import gc
import numpy as np

from tqdm import tqdm

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.schedulers import DDIMScheduler

import cross_attention


class Model:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.controlnet_attn_proc = cross_attention.CrossFrameAttnProcessor(
            unet_chunk_size=2
        )

        self.pipe = None
        self.initialized = False

    def init_model(self, use_cf_attn):
        if self.initialized:
            return

        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        self.set_model(
            model_id="runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        if use_cf_attn:
            self.pipe.unet.set_attn_processor(processor=self.controlnet_attn_proc)
            self.pipe.controlnet.set_attn_processor(processor=self.controlnet_attn_proc)

        self.initialized = True

    def set_model(self, model_id: str, **kwargs):
        if self.pipe:
            del self.pipe

        torch.cuda.empty_cache()
        gc.collect()
        safety_checker = kwargs.pop("safety_checker", None)
        self.pipe = (
            StableDiffusionControlNetPipeline.from_pretrained(
                model_id, safety_checker=safety_checker, **kwargs
            )
            .to(self.device)
            .to(self.dtype)
        )

    def run_model(self, prompt, negative_prompt, **kwargs):
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=self.generator,
            **kwargs,
        ).images

    def inference_chunk(self, frame_ids, **kwargs):
        if not self.pipe:
            return

        prompt = np.array(kwargs.pop("prompt"))[frame_ids].tolist()
        negative_prompt = np.array(kwargs.pop("negative_prompt", ""))[
            frame_ids
        ].tolist()

        if "latents" in kwargs:
            kwargs["latents"] = kwargs["latents"][frame_ids]
        if "image" in kwargs:
            kwargs["image"] = kwargs["image"][frame_ids]
        if "video_length" in kwargs:
            kwargs["video_length"] = len(frame_ids)

        return self.run_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs,
        )

    def inference(self, split_to_chunks=False, chunk_size=8, **kwargs):
        if self.pipe is None:
            return
        seed = kwargs.pop("seed", 0)
        if seed < 0:
            seed = self.generator.seed()
        kwargs.pop("generator", "")

        n_frames = (
            kwargs["image"].shape[0] if "image" in kwargs else kwargs["video_length"]
        )

        assert "prompt" in kwargs
        prompt = [kwargs.pop("prompt")] * n_frames
        negative_prompt = [kwargs.pop("negative_prompt", "")] * n_frames

        if not split_to_chunks:
            return self.run_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **kwargs,
            )

        chunk_ids = np.append(np.arange(0, n_frames, chunk_size - 1), n_frames)
        result = []
        for i in tqdm(range(len(chunk_ids) - 1), "Processing chunk"):
            frame_ids = [0] + list(range(chunk_ids[i], chunk_ids[i + 1]))
            self.generator.manual_seed(seed)
            result.append(
                self.inference_chunk(
                    frame_ids=frame_ids,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **kwargs,
                )[1:]
            )
        return np.concatenate(result)

    def video_to_video(
        self,
        video,
        control,
        prompt,
        negative_prompt,
        chunk_size=8,
        controlnet_conditioning_scale=1.0,
        num_inference_steps=20,
        guidance_scale=9.0,
        seed=42,
        eta=0.0,
        use_cf_attn=True,
    ):
        self.init_model(use_cf_attn)

        n_frames = video.shape[0]
        h, w = video.shape[2:]
        self.generator.manual_seed(seed)
        latents = torch.randn(
            (1, 4, h // 8, w // 8),
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )
        latents = latents.repeat(n_frames, 1, 1, 1)
        return self.inference(
            image=control,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=h,
            width=w,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            eta=eta,
            latents=latents,
            seed=seed,
            output_type="numpy",
            split_to_chunks=True,
            chunk_size=chunk_size,
        )

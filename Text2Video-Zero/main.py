import torch

import video_io
import video_canny
from model import Model


model = Model(device="cuda", dtype=torch.float16)


def controlnet_video_to_video(
    video_path,
    prompt,
    negative_prompt,
    chunk_size=8,
    controlnet_conditioning_scale=1.0,
    num_inference_steps=20,
    guidance_scale=9.0,
    seed=42,
    eta=0.0,
    low_threshold=100,
    high_threshold=200,
    use_cf_attn=True,
    save_path=None,
):
    video, fps = video_io.prepare_video(video_path, model.device, model.dtype)
    canny = video_canny.canny(video, low_threshold, high_threshold).to(
        model.device, model.dtype
    )
    video = model.video_to_video(
        video,
        canny,
        prompt,
        negative_prompt,
        chunk_size,
        controlnet_conditioning_scale,
        num_inference_steps,
        guidance_scale,
        seed,
        eta,
        use_cf_attn,
    )
    return video_io.create_video(video, fps, path=save_path)


def resample_video(
    video_path,
    resolution=512,
    fps=4,
    save_path=None,
):
    video, fps = video_io.prepare_video(video_path, model.device, model.dtype)
    video = video_io.resample_video(video, resolution)
    return video_io.create_video(video, fps, path=save_path)

import os

import PIL.Image
import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import cv2
from PIL import Image
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.openpose import OpenposeDetector
import decord

apply_canny = CannyDetector()
apply_openpose = OpenposeDetector()


def add_watermark(image, watermark_path, wm_rel_size=1 / 16, boundary=5):
    """
    Creates a watermark on the saved inference image.
    We request that you do not remove this to properly assign credit to
    Shi-Lab's work.
    """
    watermark = Image.open(watermark_path)
    w_0, h_0 = watermark.size
    H, W, _ = image.shape
    wmsize = int(max(H, W) * wm_rel_size)
    aspect = h_0 / w_0
    if aspect > 1.0:
        watermark = watermark.resize((wmsize, int(aspect * wmsize)), Image.LANCZOS)
    else:
        watermark = watermark.resize((int(wmsize / aspect), wmsize), Image.LANCZOS)
    w, h = watermark.size
    loc_h = H - h - boundary
    loc_w = W - w - boundary
    image = Image.fromarray(image)
    mask = watermark if watermark.mode in ("RGBA", "LA") else None
    image.paste(watermark, (loc_w, loc_h), mask)
    return image


def pre_process_canny(input_video, low_threshold=100, high_threshold=200):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, "c h w -> h w c").cpu().numpy().astype(np.uint8)
        detected_map = cv2.Canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)

    control = torch.from_numpy(detected_maps.copy()).float() / 255
    return rearrange(control, "f h w c -> f c h w")


def pre_process_pose(input_video, apply_pose_detect: bool = True):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, "c h w -> h w c").cpu().numpy().astype(np.uint8)
        img = HWC3(img)
        if apply_pose_detect:
            detected_map, _ = apply_openpose(img)
        else:
            detected_map = img
        detected_map = HWC3(detected_map)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, "f h w c -> f c h w")


def create_video(frames, fps, rescale=False, path=None):
    if path is None:
        directory = "temporal"
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "movie.mp4")

    outputs = []
    for _, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0
        x = (x * 255).numpy().astype(np.uint8)

        outputs.append(x)

    imageio.mimsave(path, outputs, fps=fps)
    return path


def prepare_video(
    video_path: str,
    resolution: int,
    device,
    dtype,
    normalize=True,
    start_t: float = 0,
    end_t: float = None,
    output_fps: int = None,
):
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()
    if not output_fps:
        output_fps = int(initial_fps)
    if not end_t:
        end_t = len(vr) / initial_fps
    else:
        end_t = min(len(vr) / initial_fps, end_t)
    assert 0 <= start_t < end_t
    assert output_fps > 0
    start_f_ind = int(start_t * initial_fps)
    end_f_ind = int(end_t * initial_fps)
    num_f = int((end_t - start_t) * output_fps)
    sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
    video = vr.get_batch(sample_idx)
    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()
    else:
        video = video.asnumpy()
    _, h, w, _ = video.shape
    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video).to(device).to(dtype)
    if h > w:
        w = int(w * resolution / h)
        w -= w % 8
        h = resolution - resolution % 8
    else:
        h = int(h * resolution / w)
        h -= h % 8
        w = resolution - resolution % 8
    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(
        video
    )
    if normalize:
        video = video / 127.5 - 1.0
    return video, output_fps


class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if not encoder_hidden_states:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length

            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")

            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

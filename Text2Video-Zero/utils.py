import os

import numpy as np
import cv2

import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import decord


def lerp(a, b, alpha):
    return a * (1 - alpha) + b * alpha


def to_uint(arr):
    return (arr * 255).astype(np.uint8)


def make_rgb(arr):
    assert arr.dtype == np.uint8
    if arr.ndim == 2:
        arr = arr[:, :, None]
    assert arr.ndim == 3
    channel = arr.shape[2]
    assert channel in [1, 3, 4]

    if channel == 1:
        arr = np.concatenate((arr,) * 3, axis=2)
    elif channel == 4:
        color = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3].astype(np.float32) / 255
        arr = np.clip(lerp(255, color, alpha), 0, 255).astype(np.uint8)
    return arr


def pre_process_canny(input_video, low_threshold=100, high_threshold=200):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, "c h w -> h w c").cpu().numpy().astype(np.uint8)
        detected_map = cv2.Canny(img, low_threshold, high_threshold)
        detected_map = make_rgb(detected_map)

        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)

    control = torch.from_numpy(detected_maps.copy()).float() / 255
    return rearrange(control, "f h w c -> f c h w")


def create_video(frames, fps, rescale=False, path=None):
    if path is None:
        directory = "temporal"
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "movie.mp4")

    outputs = []
    for k in frames:
        frame = torchvision.utils.make_grid(torch.Tensor(k), nrow=4)
        if rescale:
            frame = (frame + 1) / 2

        outputs.append(to_uint(frame.numpy()))

    imageio.mimsave(path, outputs, fps=fps)
    return path


def prepare_video(
    video_path: str,
    resolution: int,
    device,
    dtype,
    normalize=True,
    start_time: float = 0,
    end_time: float = None,
    output_fps: int = None,
):
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()

    if not output_fps:
        output_fps = int(initial_fps)

    length = len(vr) / initial_fps
    end_time = min(length, end_time) if end_time else length

    assert 0 <= start_time < end_time
    assert output_fps > 0

    start_frame_index = int(start_time * initial_fps)
    end_frame_index = int(end_time * initial_fps)
    n_frames = int((end_time - start_time) * output_fps)

    sample_idx = np.linspace(
        start_frame_index, end_frame_index, n_frames, endpoint=False
    ).astype(np.int32)
    video = vr.get_batch(sample_idx)

    video = video.detach().cpu().numpy() if torch.is_tensor(video) else video.asnumpy()

    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video).to(device, dtype)

    # resample to resolution
    hw = np.array(video.shape[2:], np.int32)
    hw = (hw // (np.amax(hw) / resolution)).astype(np.int32)
    hw -= hw % 8

    video = Resize(
        (hw[0], hw[1]), interpolation=InterpolationMode.BILINEAR, antialias=True
    )(video)

    if normalize:
        video = video / 127.5 - 1.0

    return video, output_fps


class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        is_cross_attention = encoder_hidden_states is not None

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
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

import os

import numpy as np

import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import decord


def to_uint(arr):
    return (arr * 255).astype(np.uint8)


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
    device,
    dtype,
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

    return video, output_fps


def resample_video(video, resolution: int, max_mode: bool = False):
    hw = np.array(video.shape[2:], np.int32)

    scale = np.amax(hw)
    if max_mode:
        scale = np.amin(hw)
    scale = scale / resolution

    hw = (hw // scale).astype(np.int32)
    hw -= hw % 8

    video = Resize(
        (hw[0], hw[1]), interpolation=InterpolationMode.BILINEAR, antialias=True
    )(video)

    return video

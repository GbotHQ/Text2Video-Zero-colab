import numpy as np
import cv2
import torch
from einops import rearrange


def lerp(a, b, alpha):
    return a * (1 - alpha) + b * alpha


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


def canny(input_video, low_threshold=100, high_threshold=200):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, "c h w -> h w c").cpu().numpy().astype(np.uint8)
        detected_map = cv2.Canny(img, low_threshold, high_threshold)
        detected_map = make_rgb(detected_map)

        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)

    control = torch.from_numpy(detected_maps.copy()).float() / 255
    return rearrange(control, "f h w c -> f c h w")

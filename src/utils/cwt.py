import os
from typing import List
import numpy as np
import pywt
from skimage.transform import resize
from sklearn.preprocessing import minmax_scale
from skimage.io import imsave, imread


def compute_scalogram(signal: np.ndarray, wavelet: str = "morl", scales: np.ndarray = None,
                      target_size: tuple = (224,)) -> np.ndarray:
    """Compute a 1D scalogram of a signal and resize it.

    Parameters
    ----------
    signal : np.ndarray
        1D signal of length 2048.
    wavelet : str, optional
        Wavelet name, by default "morl".
    scales : np.ndarray, optional
        Scales for the CWT. If ``None`` the range ``1..64`` is used.
    target_size : tuple, optional
        Final width after interpolation, by default ``(224,)``.
    Returns
    -------
    np.ndarray
        Vector of length ``target_size[0]`` with normalized energy.
    """
    if scales is None:
        scales = np.arange(1, 65)
    coefs, _ = pywt.cwt(signal, scales, wavelet)
    scalogram = np.abs(coefs)
    mean_energy = scalogram.mean(axis=0)
    mean_energy = minmax_scale(mean_energy)
    mean_energy_resized = resize(mean_energy, target_size, mode="reflect", preserve_range=True)
    return mean_energy_resized


def generate_cwt_image(signal_array: np.ndarray, sample_rate: int = 2048,
                        target_width: int = 224) -> List[np.ndarray]:
    """Generate a list of 2D scalogram images from a multi-channel signal.

    ``signal_array`` is expected to be shaped ``(10, N)`` and each image
    corresponds to a 1s window of 2048 samples for all 10 channels.
    """
    total_points = signal_array.shape[1]
    margin = int(0.05 * total_points)
    useful_signal = signal_array[:, margin: total_points - margin]
    win_size = sample_rate
    num_segments = useful_signal.shape[1] // win_size
    images = []
    for k in range(num_segments):
        segment = useful_signal[:, k * win_size:(k + 1) * win_size]
        image = [compute_scalogram(segment[i], target_size=(target_width,)) for i in range(segment.shape[0])]
        images.append(np.array(image))
    return images


def resize_vertical_all_repeat(images_cwt: np.ndarray, target_height: int = 224) -> np.ndarray:
    """Resize a CWT image vertically by repeating rows."""
    current_height, width = images_cwt.shape
    reps = target_height // current_height
    extra = target_height % current_height
    repeated = np.repeat(images_cwt, reps, axis=0)
    if extra > 0:
        padding = images_cwt[:extra, :]
        repeated = np.vstack([repeated, padding])
    return repeated


def save_images(images: List[np.ndarray], output_dir: str, test_name: str, classe: str) -> None:
    """Denoise and save CWT images using the global PyTorch ``model``."""

    import torch  # imported lazily to avoid mandatory dependency for other utils

    class_dir = os.path.join(output_dir, classe)
    os.makedirs(class_dir, exist_ok=True)

    for i, img in enumerate(images):
        # ``images`` are 10x224 arrays – resize vertically if needed
        if img.shape[0] != 224:
            img = resize_vertical_all_repeat(img, target_height=224)

        # normalize to [0, 1]
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        # prepare tensor [1, 1, 224, 224]
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            sigma = torch.full_like(tensor, 15 / 255, device=device)
            denoised = model(tensor, sigma)

        denoised_img = denoised.squeeze().cpu().numpy()
        img_uint8 = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)

        filename = f"{test_name.replace('/', '_')}_{i:02d}.png"
        path = os.path.join(class_dir, filename)
        imsave(path, img_uint8)

import math
import os
import logging
import torch

from .utils import get_test_result_output_dir
from brainseg.nifti_utils import quick_save_nifti_from_torch
from brainseg.scale_label_image import gaussian_kernel_3d_fft

logger = logging.getLogger("gaussian_examples_test")


def calculate_energy(tensor: torch.Tensor) -> float:
    return torch.sum(tensor ** 2).real.item()


def test_impulse_response():
    """
    Simple test function used to debug the gaussian filter impulse response.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (21, 21, 21)
    sigma = 2.0

    impulse = torch.zeros(shape, device=device)
    impulse[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1.0
    impulse_fft = torch.fft.fftn(impulse)

    # Generate a 3D Gaussian kernel in the spatial domain
    coords = [torch.arange(s, device=device) - (s - 1) / 2 for s in shape]
    z, y, x = torch.meshgrid(*coords, indexing='ij')
    squared_dist = x ** 2 + y ** 2 + z ** 2
    scaling = 1.0 / (2 * math.sqrt(2) * math.sqrt(torch.pi ** 3) * sigma ** 3)
    gaussian =  scaling * torch.exp(-squared_dist / (2 * sigma ** 2))

    kernel_fft = gaussian_kernel_3d_fft(shape, sigma, device=device)
    filtered_fft = impulse_fft * kernel_fft
    filtered = torch.fft.ifftn(filtered_fft)

    logger.info(f"Difference of filtered image and analytical solution:"
                f" {torch.sum(torch.abs(filtered - gaussian)).item()}")
    logger.info(f"Impulse Sum: {impulse.sum().item()}")
    logger.info(f"Gaussian Sum: {gaussian.sum().item()}")
    logger.info(f"Gaussian Energy: {calculate_energy(gaussian)}")
    logger.info(f"Kernel FFT Energy: {calculate_energy(kernel_fft)}")
    logger.info(f"Kernel FFT Inv Energy: {calculate_energy(torch.fft.ifftn(kernel_fft))}")
    logger.info(f"Filtered Energy: {calculate_energy(filtered)}")
    logger.info(f"Filtered Sum: {filtered.real.sum().item()}")

    output_dir = get_test_result_output_dir("gaussian_impulse")
    quick_save_nifti_from_torch(gaussian.real, os.path.join(output_dir, "gaussian.nii"))
    quick_save_nifti_from_torch(torch.fft.ifftn(kernel_fft).real, os.path.join(output_dir, "kernel.nii"))
    quick_save_nifti_from_torch(impulse_fft.abs(), os.path.join(output_dir, "impulse_fft.nii"))
    quick_save_nifti_from_torch(impulse_fft.imag, os.path.join(output_dir, "impulse_fft_imag.nii"))
    quick_save_nifti_from_torch(impulse_fft.real, os.path.join(output_dir, "impulse_fft_real.nii"))
    quick_save_nifti_from_torch(filtered.real, os.path.join(output_dir, "filtered.nii"))
    quick_save_nifti_from_torch(impulse, os.path.join(output_dir, "impulse.nii"))

import os
import logging
import torch

from .utils import get_test_result_output_dir
from brainseg.nifti_utils import quick_save_nifti_from_torch
from brainseg.scale_label_image import gaussian_filter_fft, gaussian_kernel_3d_fft

# Configure logging to output all levels to console
logger = logging.getLogger("gaussian_examples_test")

def test_impulse_response():
    """
    Impulse Input Test: Verify that the filter correctly smooths a 3D impulse signal.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (17, 17, 17)
    sigma = 2.0
    # Create an impulse delta function tensor
    impulse = torch.zeros(shape, device=device)
    impulse[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1.0

    tensor_fft = torch.fft.fftn(impulse)
    kernel_fft = gaussian_kernel_3d_fft(shape, sigma, device=device)
    kernel_fft /= kernel_fft.sum()
    filtered_fft = tensor_fft * kernel_fft
    filtered = torch.fft.ifftn(filtered_fft).real

    # Generate a 3D Gaussian kernel in the spatial domain
    coords = [torch.arange(s, device=device) - (s - 1) / 2 for s in shape]
    z, y, x = torch.meshgrid(*coords, indexing='ij')
    squared_dist = x ** 2 + y ** 2 + z ** 2
    gaussian = torch.exp(-squared_dist / (2 * sigma ** 2))
    gaussian /= gaussian.sum()

    logger.info(f"Impulse Energy: {torch.sum(impulse)}")
    logger.info(f"Gaussian Energy: {torch.sum(gaussian)}")
    logger.info(f"Filtered Energy: {torch.sum(filtered)}")

    output_dir = get_test_result_output_dir("gaussian_impulse")
    quick_save_nifti_from_torch(kernel_fft, os.path.join(output_dir, "kernel_fft.nii"))
    quick_save_nifti_from_torch(torch.fft.ifftn(kernel_fft).real, os.path.join(output_dir, "kernel.nii"))
    quick_save_nifti_from_torch(tensor_fft.real, os.path.join(output_dir, "impulse_fft.nii"))
    quick_save_nifti_from_torch(filtered, os.path.join(output_dir, "filtered.nii"))
    quick_save_nifti_from_torch(impulse, os.path.join(output_dir, "impulse.nii"))
    quick_save_nifti_from_torch(gaussian, os.path.join(output_dir, "gaussian.nii"))

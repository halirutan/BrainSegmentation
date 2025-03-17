import time

import torch
import nibabel as nib
import numpy as np
import os
from typing import Union, List, Optional
from dataclasses import dataclass

from simple_parsing import ArgumentParser
import logging
import tqdm

logger = logging.getLogger("Resample")

def gaussian_kernel_3d_fft(
        shape: tuple[int, int, int] | torch.Size,
        sigma: float,
        device: str | torch.device='cpu') -> torch.Tensor:
    """
        Generates a 3D Gaussian kernel in the frequency domain using FFT.

        This function computes a 3D Gaussian kernel directly in the frequency domain.
        It takes the shape of the desired kernel, the standard deviation (sigma),
        and the device where the computation should be performed.
        The Gaussian kernel in the frequency domain is computed based on the squared Euclidean distance
        and the provided sigma.

        Args:
            shape: The shape of the 3D Gaussian kernel, defined as (nz, ny, nx), where nz, ny, and nx are the number of
                elements along the z, y, and x dimensions respectively.
            sigma: The standard deviation of the Gaussian distribution.
                It determines the spread of the Gaussian kernel.
            device: The device where the computation should be performed,
                either 'cpu' or 'cuda'. The default is 'cpu'.

        Returns:
            torch.Tensor: A 3D tensor representing the Gaussian kernel in the
            frequency domain. The tensor has the same shape as the input `shape`.
    """
    if not sigma > 0.0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    nz, ny, nx = shape
    z = torch.fft.fftfreq(nz, device=device).view(nz, 1, 1)
    y = torch.fft.fftfreq(ny, device=device).view(1, ny, 1)
    x = torch.fft.fftfreq(nx, device=device).view(1, 1, nx)
    squared_dist = (x ** 2 + y ** 2 + z ** 2)
    kernel_fft = torch.exp(-2 * (torch.pi ** 2) * (sigma ** 2) * squared_dist / (nx * ny * nz))
    return kernel_fft


def filter_fft(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Filters a 3D tensor using a Gaussian kernel in the frequency domain.

    This function performs filtering of a 3-dimensional input tensor using a Gaussian kernel in the Fourier domain.
    The input tensor is transformed into the frequency domain via FFT, multiplied element-wise with the FFT of the
    Gaussian kernel, and then transformed back to the spatial domain using the inverse FFT.

    Args:
        tensor: A 3D PyTorch tensor to be filtered. The tensor must have three dimensions (ndim == 3).
        sigma: The standard deviation of the Gaussian kernel used for filtering.

    Returns:
        torch.Tensor: The filtered tensor in the spatial domain.
    """
    assert tensor.ndim == 3, "Input tensor must be 3D"

    device = tensor.device
    shape = tensor.shape

    tensor_fft = torch.fft.fftn(tensor)
    kernel_fft = gaussian_kernel_3d_fft(shape, sigma, device=device)
    filtered_fft = tensor_fft * kernel_fft
    filtered = torch.fft.ifftn(filtered_fft).real
    return filtered

def smooth_label_image(data: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Smooth a label image using a Gaussian kernel of size sigma.

    This function applies Gaussian smoothing to a label image tensor and assigns each label the
    maximum probability using a Gaussian Kernel. It processes each unique label separately, applies
    3D FFT-based Gaussian smoothing, and updates the resulting tensor with the labels based on
    maximum probability.

    Args:
        data: A tensor containing the label image to be smoothed.
        sigma: Standard deviation for the Gaussian kernel used for smoothing.

    Returns:
        torch.Tensor: A tensor representing the smoothed label image.
    """
    labels = torch.unique(data)
    kernel_fft = gaussian_kernel_3d_fft(data.shape, sigma, device=data.device)
    max_probabilities = torch.zeros_like(data, dtype=torch.float32)
    result = torch.zeros_like(data)
    logger.info(f"Smoothing {len(labels)} labels")
    tqd = tqdm.tqdm(labels, desc="Smoothing label")
    for label in tqd:
        tqd.set_postfix({"label": label})
        filtered = torch.fft.ifftn(torch.fft.fftn((data == label).float()) * kernel_fft).real
        mask2 = filtered > max_probabilities
        result[mask2] = label
        max_probabilities[mask2] = filtered[mask2]
    return result


def resample_label_image(nifti: nib.Nifti1Image,
                         resolution_out: Union[float, List[float]],
                         sigma: Optional[float] = None,
                         device: str | torch.device = "cpu") -> nib.Nifti1Image:
    """
    Resample a label image to a new resolution.
    Note: This is intended for internal use only to upsample label images that are in a too low resolution.

    Args:
        nifti: The input label-image represented as a nib.Nifti1Image object.
        resolution_out: The desired output resolution. It can be a single float value or a list of three float values
            representing the desired voxel size in mm in x, y, and z directions respectively.
        sigma: If not None, it must be a float value specifying the standard deviation of the Gaussian kernel to be
            used for smoothing the label image.
    Returns:
        The resampled label-image represented as a nib.Nifti1Image object.
    """
    if type(resolution_out) is float:
        resolution_out = np.array([resolution_out] * 3)
    else:
        resolution_out = np.array(resolution_out)
    data_resampled = do_resample(nifti, resolution_out)
    if sigma is not None:
        data_resampled = smooth_label_image(data_resampled, sigma)
    header = nifti.header
    # noinspection PyUnresolvedReferences
    header.set_zooms(resolution_out)
    # noinspection PyTypeChecker
    return nib.Nifti1Image(data_resampled.numpy(force=True).astype(np.uint16), nifti.affine, header)


def do_resample(
        nifti: nib.Nifti1Image,
        resolution_out: np.ndarray,
        device: str | torch.device = "cpu") -> torch.Tensor:
    header = nifti.header
    # noinspection PyUnresolvedReferences
    dim: tuple[int, int, int] = header.get_data_shape()
    if len(dim) != 3:
        raise RuntimeError("Image data does not have 3 dimensions")

    # noinspection PyUnresolvedReferences
    resolution_in = header["pixdim"][1:4]
    step_size: np.ndarray[np.float32] = resolution_out / resolution_in
    zs = torch.arange(0, dim[0], step_size[0]).to(dtype=torch.int, device=device)
    ys = torch.arange(0, dim[1], step_size[1]).to(dtype=torch.int, device=device)
    xs = torch.arange(0, dim[2], step_size[2]).to(dtype=torch.int, device=device)
    numpy_data = nifti.get_fdata()
    torch_data = torch.tensor(numpy_data, dtype=torch.int, device=device)
    data_resampled = torch_data[torch.meshgrid(zs, ys, xs, indexing="ij")]
    return data_resampled


@dataclass
class RescaleLabelImageData:
    image_file: str
    """Input label image for rescaling."""

    output_dir: str
    """
    Output directory where to store the resampled image.
    """

    resolution: float
    """
    Resolution in mm for the resampled label image.
    """

    sigma: Optional[float] = None
    """If not None, it must be a float value specifying the standard deviation of the Gaussian kernel to be used for smoothing the label image."""


def main():
    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(RescaleLabelImageData, "general")
    args = parser.parse_args()
    options: RescaleLabelImageData = args.general

    if isinstance(options.output_dir, str) and os.path.isdir(options.output_dir):
        logger.debug(f"Using output directory: '{options.output_dir}'")
    else:
        logger.error(f"Output directory does not exist: '{options.output_dir}'")
        exit(1)

    if isinstance(options.image_file, str) and os.path.isfile(options.image_file):
        logger.debug(f"Using label image: '{options.image_file}'")
    else:
        logger.error(f"Provided image is not a regular file: '{options.image_file}'")
        exit(1)

    resolution = options.resolution
    output_dir = options.output_dir

    image_file = options.image_file
    nifti = nib.load(image_file)
    if not isinstance(nifti, nib.Nifti1Image):
        logger.error(f"Image {image_file} is not a Nifti1 image")
        exit(1)

    result = resample_label_image(nifti, resolution)

    if not isinstance(result, nib.Nifti1Image):
        logger.error("Unable to rescale image.")

    file_base = os.path.basename(image_file)
    output_file = os.path.join(output_dir, file_base)
    nib.save(result, output_file)


def select_labels(img: nib.Nifti1Image, labels: List[int]) -> nib.Nifti1Image:
    data = img.get_fdata().astype(np.uint16)
    new_data = np.zeros_like(data, dtype=np.uint16)
    for label in labels:
        new_data[data == label] = label
    # noinspection PyTypeChecker
    return nib.Nifti1Image(new_data, img.affine, img.header)

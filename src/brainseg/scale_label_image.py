import time

import torch
import nibabel as nib
import numpy as np
import os
from typing import Union, List, Optional, Any
from dataclasses import dataclass

from nibabel.arrayproxy import ArrayLike
from numpy import ndarray, dtype, floating
from numpy._core.numeric import _SCT
from simple_parsing import ArgumentParser
import logging
from torch.nn.functional import conv1d
import tqdm


def gaussian_kernel_3d_fft(shape, sigma, device='cpu'):
    """Creates a 3D Gaussian kernel in the frequency domain using FFT."""
    nz, ny, nx = shape
    z = torch.fft.fftfreq(nz, device=device).view(nz, 1, 1)
    y = torch.fft.fftfreq(ny, device=device).view(1, ny, 1)
    x = torch.fft.fftfreq(nx, device=device).view(1, 1, nx)
    squared_dist = (x ** 2 + y ** 2 + z ** 2)
    kernel_fft = torch.exp(-2 * (torch.pi ** 2) * (sigma ** 2) * squared_dist)
    return kernel_fft


def filter_fft(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Applies a 3D Gaussian filter using FFT."""
    assert tensor.ndim == 3, "Input tensor must be 3D"

    device = tensor.device
    shape = tensor.shape

    # Compute FFT of the input tensor
    tensor_fft = torch.fft.fftn(tensor)

    # Compute FFT of the Gaussian kernel
    kernel_fft = gaussian_kernel_3d_fft(shape, sigma, device=device)

    # Multiply in the frequency domain
    filtered_fft = tensor_fft * kernel_fft

    # Inverse FFT to return to the spatial domain
    filtered = torch.fft.ifftn(filtered_fft).real  # Take only real part

    return filtered


def gaussian_kernel_1d(sigma: float, device: Union[str, torch.device]='cpu', dtype=torch.float32):
    """Creates a 1D Gaussian kernel given a standard deviation `sigma`."""
    kernel_size = int(2 * (3.0 * sigma) + 1)  # Approximate kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size

    coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g /= g.sum()
    return g


def filter_separable(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Applies a 3D Gaussian filter using three separable 1D convolutions."""
    assert tensor.ndim == 3, "Input tensor must be 3D"
    assert tensor.device == kernel.device, "Input and kernel must be on the same device"
    assert kernel.ndim == 1, "Kernel must be 1D"

    k1d = kernel.view(1, 1, -1).to(tensor.dtype)
    for _ in range(3):
        tensor = conv1d(tensor.reshape(-1, 1, tensor.size(2)), k1d, padding="same").view(*tensor.shape)
        tensor = tensor.permute(2, 0, 1)
    return tensor


logger = logging.getLogger("Resample")

def smooth_label_image(data: torch.Tensor, sigma: float) -> torch.Tensor:
    labels = torch.unique(data)
    # kernel = gaussian_kernel_1d(sigma, device=data.device, dtype=torch.float32)
    kernel_fft = gaussian_kernel_3d_fft(data.shape, sigma, device=data.device)
    max_probabilities = torch.zeros_like(data, dtype=torch.float32)
    # kern = torch.randn(data.shape, device=data.device, dtype=torch.float32)
    result = torch.zeros_like(data)
    tqd = tqdm.tqdm(labels, desc="Processing labels")
    print(labels)
    for label in tqd:
        filtered = torch.fft.ifftn(torch.fft.fftn((data == label).float()) * kernel_fft).real
        mask2 = filtered > max_probabilities
        result[mask2] = label
        max_probabilities[mask2] = filtered[mask2]
    return result


def resample_label_image(nifti: nib.Nifti1Image,
                         resolution_out: Union[float, List[float]],
                         sigma: Optional[float] = None) -> nib.Nifti1Image:
    """
    Resample a label image to a new resolution.
    Note: This is intended for internal use only to upsample label images that are in a too low resolution.

    Args:
        nifti: The input label image represented as a nib.Nifti1Image object.
        resolution_out: The desired output resolution. It can be a single float value or a list of three float values
            representing the desired voxel size in mm in x, y, and z directions respectively.
        sigma: If not None, it must be a float value specifying the standard deviation of the Gaussian kernel to be
            used for smoothing the label image.
    Returns:
        The resampled label image represented as a nib.Nifti1Image object.
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
    return nib.Nifti1Image(data_resampled.numpy(force=True).astype(np.uint16), nifti.affine, header)

def do_resample(nifti: nib.Nifti1Image, resolution_out: np.ndarray) -> torch.Tensor:
    header = nifti.header
    # noinspection PyUnresolvedReferences
    dim: tuple[int, int, int] = header.get_data_shape()
    if len(dim) != 3:
        raise RuntimeError("Image data does not have 3 dimensions")

    device = "mps"
    # noinspection PyUnresolvedReferences
    resolution_in = header["pixdim"][1:4]
    step_size: np.ndarray[np.float32] = resolution_out / resolution_in
    zs = torch.arange(0, dim[0], step_size[0]).to(dtype=torch.int, device=device)
    ys = torch.arange(0, dim[1], step_size[1]).to(dtype=torch.int, device=device)
    xs = torch.arange(0, dim[2], step_size[2]).to(dtype=torch.int, device=device)
    numpy_data = nifti.get_fdata()
    torch_data = torch.tensor(numpy_data, dtype=torch.int, device=device)
    start = time.time()
    data_resampled = torch_data[torch.meshgrid(zs, ys, xs, indexing="ij")]
    end = time.time()
    print(f"Resampling took {end - start} seconds")
    return data_resampled


@dataclass
class Options:
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

    crop_size: int = None
    """If not None, it must be an int or a list of integers specifying the output size."""


def main():
    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(Options, "general")
    args = parser.parse_args()
    options: Options = args.general

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
    crop_size = options.crop_size
    output_dir = options.output_dir

    image_file = options.image_file
    nifti = nib.load(image_file)
    if not isinstance(nifti, nib.Nifti1Image):
        logger.error(f"Image {image_file} is not a Nifti1 image")
        exit(1)

    if crop_size is not None:
        result = resample_label_image_cropped(nifti, resolution, crop_size)
    else:
        result = resample_label_image(nifti, resolution)

    if not isinstance(result, nib.Nifti1Image):
        logger.error("Unable to rescale image.")

    file_base = os.path.basename(image_file)
    output_file = os.path.join(output_dir, file_base)
    nib.save(result, output_file)

def select_labels(img: nib.Nifti1Image, labels: List[int]) -> nib.Nifti1Image:
    data = img.get_fdata()
    new_data = np.zeros_like(data, dtype=np.uint16)
    for label in labels:
        new_data[data == label] = label
    return nib.Nifti1Image(new_data, img.affine, img.header)

def test_filter():
    device = "cpu"
    img_file = "/Users/pscheibe/PycharmProjects/SynthSeg/data/training_label_maps/training_seg_01.nii.gz"
    img = nib.load(img_file)
    data = torch.from_numpy(img.get_fdata()).to(device=device, dtype=torch.float32)
    kernel = gaussian_kernel_1d(4.0).to(device=device)
    data_filtered = filter_separable(data, kernel)
    nib.save(nib.Nifti1Image(data_filtered, img.affine, img.header), "/Users/pscheibe/tmp/test.nii.gz")


if __name__ == '__main__':
    torch.set_num_threads(256)
    img_file = "/Users/pscheibe/PycharmProjects/SynthSeg/data/training_label_maps/training_seg_05.nii.gz"
    for sigma in [None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
    # for sigma in [6.0, 7.0]:
        img = nib.load(img_file)
        img_out = resample_label_image(img, [0.4, 0.4, 0.4], sigma=sigma)
        img_selected = select_labels(img_out, [41, 2])
        nib.save(img_selected, f"/Users/pscheibe/tmp/test_{sigma}.nii.gz")
    # img_out = resample_label_image(img, [0.5, 0.5, 0.5], sigma=None)
    # nib.save(img_out, "/Users/pscheibe/tmp/test_None.nii.gz")



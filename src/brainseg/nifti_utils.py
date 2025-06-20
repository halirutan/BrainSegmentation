"""
This module provides utility functions for working with NIfTI images and PyTorch tensors, 
focusing on easy saving and manipulation of medical imaging data in the NIfTI format.
"""

from typing import Optional, Tuple, List

import torch
import numpy as np
import nibabel as nib
from nibabel import Nifti1Image



def quick_save_nifti_from_torch(
        tensor: torch.Tensor,
        output_path: str,
        dtype: Optional[np.dtype] = None
) -> None:
    """
    Save a PyTorch tensor as a Nifti image with a field of view from 0 to 1 in all axes.

    Args:
        tensor: A PyTorch tensor to be saved as a Nifti image. Expected to be 3D.
        output_path: Path where the Nifti image will be saved.
        dtype: Data type for the saved image.
            If None, it will use np.float32 for float tensors and np.int16 for integer tensors.

    Returns:
        None
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tensor.ndim}D tensor")

    if tensor.is_cuda:
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor.detach().numpy()
    quick_save_nifti_from_numpy(array, output_path, dtype)


def quick_save_nifti_from_numpy(
        array: np.ndarray,
        output_path: str,
        dtype: Optional[np.dtype] = None
) -> None:
    """
    Save a NumPy array as a Nifti image with a field of view from 0 to 1 in all axes.

    Args:
        array: A NumPy array to be saved as a Nifti image. Expected to be 3D.
        output_path: Path where the Nifti image will be saved.
        dtype: Data type for the saved image. If None, will use np.float32 for float arrays
               and np.int16 for integer arrays.

    Returns:
        None
    """
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {array.ndim}D array")

    if dtype is None:
        if np.issubdtype(array.dtype, np.integer):
            dtype = np.int32
        else:
            dtype = np.float32

    affine = _create_affine_for_unit_fov(array.shape)
    # noinspection PyTypeChecker
    nifti_img = nib.Nifti1Image(array.astype(dtype), affine)
    nib.save(nifti_img, output_path)


def _create_affine_for_unit_fov(shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Create an affine transformation matrix for a field of view from 0 to 1 in all axes.

    Args:
        shape: Shape of the 3D array (z, y, x).

    Returns:
        4x4 affine transformation matrix as a NumPy array.
    """
    voxel_sizes = np.array([1.0 / dim for dim in shape])
    affine = np.zeros((4, 4))
    affine[0, 0] = voxel_sizes[2]
    affine[1, 1] = voxel_sizes[1]
    affine[2, 2] = voxel_sizes[0]
    affine[3, 3] = 1.0

    return affine


def is_label_image(img: Nifti1Image) -> bool:
    """
     Determine if the given image is a labeled image.

     A labeled image is typically represented with an integer data type, which
     corresponds to discrete labels for different regions or objects within the
     image.

     Args:
         img (Nifti1Image): An image object of type `Nifti1Image` to check.

     Returns:
         bool: True if the image is a labeled image (i.e., its data type is an integer), False otherwise.
     """
    return np.issubdtype(img.get_data_dtype(), np.integer)


def select_labels(img: nib.Nifti1Image, labels: List[int]) -> nib.Nifti1Image:
    """
    Select specific labels from a NIfTI label image and create a new image containing
    only those labels.

    Args:
        img (nib.Nifti1Image): The input NIfTI image, which must be a label image.
        labels (List[int]): A list of integer labels to select from the image.

    Returns:
        nib.Nifti1Image: A new NIfTI image containing the selected labels, with other regions set to background.

    Raises:
        ValueError: If the provided image is not a label image.
    """
    if not is_label_image(img):
        raise ValueError("Image is not a label image")
    data_type = img.get_data_dtype()
    data = img.get_fdata(dtype=data_type)
    new_data = np.zeros_like(data, dtype=data_type)
    for label in labels:
        new_data[data == label] = label
    # noinspection PyTypeChecker
    return nib.Nifti1Image(new_data, img.affine, img.header)

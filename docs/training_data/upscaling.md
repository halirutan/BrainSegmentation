# Upscaling Label Images

This document provides a guide on how to use the upscaling functionality implemented in the BrainSegmentation library.

## Overview

The upscaling functionality allows you to resample 3D label images (in NIfTI format) to a higher resolution. This is particularly useful when you have label images that are in a lower resolution than desired for your analysis or visualization.

The upscaling process consists of two main steps:
1. **Resampling**: The label image is resampled to the desired resolution using nearest neighbor interpolation.
2. **Optional Smoothing**: If specified, a Gaussian smoothing is applied to the resampled image to reduce artifacts and improve the quality of the upscaled image.

## How It Works

### Resampling

The resampling process uses nearest neighbor interpolation to resize the image to the desired resolution. This is done by:
1. Calculating the step size based on the ratio of the output resolution to the input resolution
2. Creating a grid of points in the new resolution space
3. Sampling the original image at these points using nearest neighbor interpolation

### Smoothing (Optional)

If a sigma value is provided, the upscaled image undergoes Gaussian smoothing:
1. Each unique label in the image is processed separately
2. A 3D Gaussian kernel is applied in the frequency domain using Fast Fourier Transform (FFT)
3. The smoothed labels are combined based on maximum probability

This smoothing helps to create more natural-looking boundaries between different labels in the upscaled image.

## Usage

### Command-Line Interface

The upscaling functionality can be used directly from the command line:

```bash
python -m brainseg.scale_label_image \
    --image_file /path/to/input/label_image.nii \
    --output_dir /path/to/output/directory \
    --resolution 0.5 \
    [--sigma 0.5]
```

Parameters:
- `--image_file`: Path to the input label image (NIfTI format)
- `--output_dir`: Directory where the resampled image will be saved
- `--resolution`: Desired output resolution in mm (e.g., 0.5 for 0.5mm isotropic resolution)
- `--sigma` (optional): Standard deviation for the Gaussian kernel used for smoothing

### Programmatic Usage

You can also use the upscaling functionality in your Python code:

```python
import nibabel as nib
from brainseg.scale_label_image import resample_label_image

# Load the input label image
input_image = nib.load('/path/to/input/label_image.nii')

# Resample to 0.5mm resolution
resampled_image = resample_label_image(
    nifti=input_image,
    resolution_out=0.5,  # Can also be a list [0.5, 0.5, 0.5] for anisotropic resolution
    sigma=0.5,  # Optional, set to None to skip smoothing
    device="cpu"  # Use "cuda" for GPU acceleration if available
)

# Save the resampled image
nib.save(resampled_image, '/path/to/output/resampled_image.nii')
```

## Tips and Considerations

- The input image must be a 3D NIfTI image with label data
- Higher resolution (smaller voxel size) will result in larger output files
- The smoothing parameter (sigma) controls the amount of smoothing applied; larger values result in more smoothing
- GPU acceleration can significantly speed up the process for large images if you have a CUDA-compatible GPU

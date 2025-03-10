# API Reference

This page provides detailed documentation for the Brain Segmentation API.

## Module: `brainseg.scale_label_image`

This module provides functionality for scaling label images (segmentation masks) while preserving label values.

### Functions

#### `scale`

```python
def scale(label_image, new_size, method='nearest', **kwargs):
    """
    Scale a label image to a new size while preserving label values.
    
    Parameters
    ----------
    label_image : ndarray
        The input label image (segmentation mask) to be scaled.
        Can be 2D or 3D.
    
    new_size : tuple
        The target size as (height, width) for 2D or (depth, height, width) for 3D.
    
    method : str, default='nearest'
        The interpolation method to use. Options include:
        - 'nearest': Nearest neighbor interpolation (recommended for label images)
        - 'linear': Linear interpolation (may create new label values)
        - 'cubic': Cubic interpolation (may create new label values)
    
    **kwargs : dict
        Additional arguments to pass to the underlying scaling function.
        Common options include:
        - preserve_range: bool, whether to keep the original value range
        - anti_aliasing: bool, whether to apply anti-aliasing (should be False for label images)
    
    Returns
    -------
    ndarray
        The scaled label image with the same data type as the input.
    
    Notes
    -----
    When scaling label images, it's generally recommended to use 'nearest' interpolation
    to avoid creating new label values that weren't in the original image.
    
    Examples
    --------
    >>> import numpy as np
    >>> from brainseg import scale_label_image
    >>> # Create a simple 2D label image
    >>> labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
    >>> # Scale to double the size
    >>> scaled = scale_label_image.scale(labels, (8, 8), method='nearest')
    >>> scaled.shape
    (8, 8)
    """
    # Implementation details...
```

#### `scale_to_reference`

```python
def scale_to_reference(label_image, reference_image, method='nearest', **kwargs):
    """
    Scale a label image to match the dimensions of a reference image.
    
    Parameters
    ----------
    label_image : ndarray
        The input label image (segmentation mask) to be scaled.
    
    reference_image : ndarray
        The reference image whose dimensions will be matched.
    
    method : str, default='nearest'
        The interpolation method to use.
    
    **kwargs : dict
        Additional arguments to pass to the underlying scaling function.
    
    Returns
    -------
    ndarray
        The scaled label image with dimensions matching the reference image.
    """
    # Implementation details...
```

## Command Line Interface

The module can also be used as a command-line tool:

```bash
python -m brainseg.scale_label_image --help
```

### Arguments

- `--input`: Path to the input label image file (NIfTI format)
- `--output`: Path to save the output scaled image
- `--size`: Target dimensions (e.g., "256 256 256")
- `--method`: Interpolation method (default: "nearest")

### Example

```bash
python -m brainseg.scale_label_image --input segmentation.nii.gz --output scaled.nii.gz --size 128 128 128
```

## Error Handling

The module includes error handling for common issues:

- Invalid input dimensions
- Unsupported interpolation methods
- Memory errors when dealing with large volumes

For more examples and usage scenarios, see the [Examples](examples.md) page.
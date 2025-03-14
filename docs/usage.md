# Usage Guide

This guide explains how to use the Brain Segmentation tools for your projects.

## Basic Usage

The Brain Segmentation package provides tools for processing and analyzing brain images. Here's how to get started:

```python
import brainseg

# Example code will be provided here
```

## Scaling Label Images

One of the main features is the ability to scale label images. This is useful for resizing segmentation masks while preserving their categorical information:

```python
from brainseg import scale_label_image

# Load your label image
# label_image = ...

# Scale the label image to a new size
scaled_image = scale_label_image.scale(
    label_image, 
    new_size=(256, 256, 256),
    method='nearest'
)

# Save or process the scaled image
# ...
```

## Command Line Interface

Some functionality is also available through command-line interfaces:

```bash
# Example command-line usage
python -m brainseg.scale_label_image --input input.nii.gz --output output.nii.gz --size 256 256 256
```

## Configuration Options

The Brain Segmentation tools can be configured with various parameters:

- `method`: Interpolation method ('nearest', 'linear', etc.)
- `size`: Target dimensions for scaling
- Additional parameters specific to each function

Refer to the [API Reference](api_reference.md) for detailed information on all available functions and their parameters.

## Best Practices

- Pre-process your brain images for optimal results
- Use appropriate interpolation methods for your specific use case
- Validate results visually when possible
- Consider memory usage when working with large 3D volumes
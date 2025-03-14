# Examples

This page provides examples of how to use the Brain Segmentation tools for various common tasks.

## Basic Example: Scaling a Label Image

This example demonstrates how to load a label image, scale it to a new size, and save the result:

```python
import nibabel as nib
from brainseg import scale_label_image

# Load a label image (segmentation mask)
input_file = "path/to/segmentation.nii.gz"
img = nib.load(input_file)
label_data = img.get_fdata()

# Scale the label image to half its original size
original_shape = label_data.shape
new_shape = (original_shape[0]//2, original_shape[1]//2, original_shape[2]//2)

scaled_data = scale_label_image.scale(
    label_data,
    new_size=new_shape,
    method='nearest'  # Use nearest neighbor to preserve label values
)

# Create a new NIfTI image with the scaled data
scaled_img = nib.Nifti1Image(scaled_data, img.affine * 2)  # Adjust affine for new voxel size

# Save the scaled image
output_file = "path/to/output/scaled_segmentation.nii.gz"
nib.save(scaled_img, output_file)

print(f"Original shape: {original_shape}, New shape: {scaled_data.shape}")
```

## Batch Processing Multiple Images

This example shows how to process multiple label images in a directory:

```python
import os
import nibabel as nib
from brainseg import scale_label_image

def process_directory(input_dir, output_dir, scale_factor=0.5):
    """Process all NIfTI files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Load image
            img = nib.load(input_path)
            data = img.get_fdata()
            
            # Calculate new dimensions
            original_shape = data.shape
            new_shape = tuple(int(dim * scale_factor) for dim in original_shape)
            
            # Scale the image
            scaled_data = scale_label_image.scale(
                data,
                new_size=new_shape,
                method='nearest'
            )
            
            # Save the result
            scaled_img = nib.Nifti1Image(scaled_data, img.affine / scale_factor)
            nib.save(scaled_img, output_path)
            
            print(f"Processed {filename}")

# Example usage
process_directory("path/to/input/labels", "path/to/output/scaled_labels")
```

## Command Line Example

You can also use the command line interface for quick processing:

```bash
# Scale a single image
python -m brainseg.scale_label_image --input segmentation.nii.gz --output scaled.nii.gz --size 128 128 128

# Process multiple files using a shell loop
for file in input_dir/*.nii.gz; do
    output_file="output_dir/$(basename $file)"
    python -m brainseg.scale_label_image --input "$file" --output "$output_file" --size 128 128 128
done
```

## Advanced Usage: Custom Interpolation

For advanced users who need more control over the interpolation process:

```python
import numpy as np
import nibabel as nib
from brainseg import scale_label_image

# Load your label image
img = nib.load("segmentation.nii.gz")
data = img.get_fdata()

# Get unique label values
unique_labels = np.unique(data)
print(f"Image contains {len(unique_labels)} unique labels: {unique_labels}")

# Scale with custom options
scaled_data = scale_label_image.scale(
    data,
    new_size=(256, 256, 256),
    method='nearest',
    preserve_range=True,  # Ensure the output has the same value range as input
    anti_aliasing=False   # Disable anti-aliasing to preserve exact label values
)

# Verify that no new labels were created during scaling
scaled_unique_labels = np.unique(scaled_data)
print(f"Scaled image contains {len(scaled_unique_labels)} unique labels: {scaled_unique_labels}")

# Save the result
scaled_img = nib.Nifti1Image(scaled_data, img.affine)
nib.save(scaled_img, "custom_scaled.nii.gz")
```

These examples should help you get started with the Brain Segmentation tools. For more detailed information, refer to the [Usage Guide](usage.md) and [API Reference](api_reference.md).
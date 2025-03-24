import numpy as np
import nibabel as nib
from brainseg.create_fh_seg import isolate_labels, resample_to_target, verify_label_consistency_across_overlays
import pytest

def test_isolate_labels(tmpdir):
    # Create a simple 3D array with predefined labels
    data = np.array([
        [[0, 501], [999, 502]],
        [[506, 507], [508, 0]]
    ], dtype=np.int32)

    affine = np.eye(4)
    input_img = nib.Nifti1Image(data, affine)

    input_path = tmpdir.join("input.nii.gz")
    output_path = tmpdir.join("output.nii.gz")

    nib.save(input_img, str(input_path))

    # Test function: isolate specific labels
    isolate_labels(str(input_path), [501, 506, 507], str(output_path))

    output_data = nib.load(str(output_path)).get_fdata().astype(np.int32)

    expected_output = np.array([
        [[0, 501], [0, 0]],
        [[506, 507], [0, 0]]
    ], dtype=np.int32)

    assert np.array_equal(output_data, expected_output), "Label isolation failed."

def test_resample_to_target():
    source_data = np.ones((2, 2, 2), dtype=np.int32)
    target_data = np.zeros((4, 4, 4), dtype=np.int32)

    source_img = nib.Nifti1Image(source_data, affine=np.eye(4))
    target_img = nib.Nifti1Image(target_data, affine=np.eye(4))

    resampled_img = resample_to_target(source_img, target_img)
    resampled_data = resampled_img.get_fdata()

    assert resampled_data.shape == target_data.shape, "Resampled image shape mismatch."
    assert np.all(resampled_data == 1), "Resampled image data incorrect."

def test_verify_label_consistency_across_overlays(tmpdir):
    affine = np.eye(4)

    # Create consistent overlays
    overlay1 = tmpdir.join("overlay1.nii.gz")
    overlay2 = tmpdir.join("overlay2.nii.gz")

    data1 = np.array([0, 501, 502], dtype=np.int32)
    data2 = np.array([0, 502, 501], dtype=np.int32)

    nib.save(nib.Nifti1Image(data1, affine), str(overlay1))
    nib.save(nib.Nifti1Image(data2, affine), str(overlay2))

    # This should pass without errors
    verify_label_consistency_across_overlays([str(overlay1), str(overlay2)])

    # Create an inconsistent overlay with a different label
    overlay3 = tmpdir.join("overlay3.nii.gz")
    data3 = np.array([0, 999], dtype=np.int32)
    nib.save(nib.Nifti1Image(data3, affine), str(overlay3))

    # Expecting a ValueError due to inconsistent labels
    with pytest.raises(ValueError):
        verify_label_consistency_across_overlays([str(overlay1), str(overlay3)])

import os
import unittest
import tempfile
import numpy as np
import torch
import nibabel as nib

from brainseg.nifti_utils import quick_save_nifti_from_torch, quick_save_nifti_from_numpy
from brainseg.nifti_utils import _create_affine_for_unit_fov

class TestNiftiUtils(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()

        # Create sample data
        self.shape = (10, 15, 20)  # z, y, x
        self.numpy_array = np.random.rand(*self.shape)
        self.torch_tensor = torch.rand(self.shape)

        # Define output paths
        self.numpy_output_path = os.path.join(self.test_dir.name, "numpy_test.nii.gz")
        self.torch_output_path = os.path.join(self.test_dir.name, "torch_test.nii.gz")

    def tearDown(self):
        # Clean up temporary directory
        self.test_dir.cleanup()

    def test_create_affine_for_unit_fov(self):
        """Test that the affine matrix is created correctly for a unit field of view."""
        affine = _create_affine_for_unit_fov(self.shape)

        # Check shape
        self.assertEqual(affine.shape, (4, 4))

        # Check voxel sizes
        self.assertAlmostEqual(affine[0, 0], 1.0 / self.shape[2])  # x voxel size
        self.assertAlmostEqual(affine[1, 1], 1.0 / self.shape[1])  # y voxel size
        self.assertAlmostEqual(affine[2, 2], 1.0 / self.shape[0])  # z voxel size

        # Check last row
        self.assertEqual(affine[3, 3], 1.0)

    def test_save_nifti_from_numpy(self):
        """Test saving a Nifti image from a NumPy array."""
        # Save the array
        quick_save_nifti_from_numpy(self.numpy_array, self.numpy_output_path)

        # Check that the file exists
        self.assertTrue(os.path.exists(self.numpy_output_path))

        # Load the saved file
        img = nib.load(self.numpy_output_path)

        # Check data shape
        self.assertEqual(img.shape, self.shape)

        # Check affine
        expected_affine = _create_affine_for_unit_fov(self.shape)
        np.testing.assert_array_almost_equal(img.affine, expected_affine)

    def test_save_nifti_from_torch(self):
        """Test saving a Nifti image from a PyTorch tensor."""
        # Save the tensor
        quick_save_nifti_from_torch(self.torch_tensor, self.torch_output_path)

        # Check that the file exists
        self.assertTrue(os.path.exists(self.torch_output_path))

        # Load the saved file
        img = nib.load(self.torch_output_path)

        # Check data shape
        self.assertEqual(img.shape, self.shape)

        # Check affine
        expected_affine = _create_affine_for_unit_fov(self.shape)
        np.testing.assert_array_almost_equal(img.affine, expected_affine)

    def test_dtype_handling(self):
        """Test that data types are handled correctly."""
        # Integer array
        int_array = np.random.randint(0, 100, size=self.shape, dtype=np.int32)
        int_output_path = os.path.join(self.test_dir.name, "int_test.nii.gz")
        quick_save_nifti_from_numpy(int_array, int_output_path)

        # Load and check dtype
        int_img = nib.load(int_output_path)
        self.assertEqual(int_img.get_data_dtype(), np.int16)

        # Float array with explicit dtype
        float_array = np.random.rand(*self.shape)
        float_output_path = os.path.join(self.test_dir.name, "float_test.nii.gz")
        quick_save_nifti_from_numpy(float_array, float_output_path, dtype=np.float64)

        # Load and check dtype
        float_img = nib.load(float_output_path)
        self.assertEqual(float_img.get_data_dtype(), np.float64)

if __name__ == "__main__":
    unittest.main()

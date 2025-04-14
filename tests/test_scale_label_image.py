import math

import pytest
from brainseg.scale_label_image import gaussian_kernel_3d_fft, gaussian_filter_fft

import unittest
import torch

class TestGaussianFilter(unittest.TestCase):

    def setUp(self):
        # Define common parameters for the tests
        self.shape = (31, 31, 31)
        self.sigma = 2.0
        self.device = 'cpu'

    def test_gaussian_kernel_3d_fft_correct_dimensions(self):
        shape = (8, 8, 8)
        sigma = 1.0
        kernel_fft = gaussian_kernel_3d_fft(shape, sigma)
        assert kernel_fft.shape == shape, f"Expected shape {shape}, got {kernel_fft.shape}"

    def test_gaussian_kernel_3d_fft_dtype(self):
        shape = (4, 4, 4)
        sigma = 2.5
        kernel_fft = gaussian_kernel_3d_fft(shape, sigma)
        assert kernel_fft.dtype == torch.float32, "Expected kernel dtype to be torch.float32"

    def test_gaussian_kernel_3d_fft_values_not_nan(self):
        shape = (6, 6, 6)
        sigma = 1.2
        kernel_fft = gaussian_kernel_3d_fft(shape, sigma)
        assert not torch.isnan(kernel_fft).any(), "Kernel contains NaN values"

    def test_gaussian_kernel_3d_fft_on_cuda(self):
        if torch.cuda.is_available():
            shape = (10, 10, 10)
            sigma = 3.0
            device = "cuda"
            kernel_fft = gaussian_kernel_3d_fft(shape, sigma, device=device)
            assert kernel_fft.device.type == "cuda", "Kernel not on CUDA device"
            assert kernel_fft.shape == shape, f"Expected shape {shape}, got {kernel_fft.shape}"

    def test_gaussian_kernel_3d_fft_invalid_sigma(self):
        shape = (5, 5, 5)
        sigma = -1.0
        with pytest.raises(ValueError):
            gaussian_kernel_3d_fft(shape, sigma)

    def test_impulse_response(self):
        """
        Impulse Input Test: Verify that the filter correctly smooths a 3D impulse signal.
        """
        # Create an impulse delta function tensor
        impulse = torch.zeros(self.shape, device=self.device)
        impulse[self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2] = 1.0

        filtered = gaussian_filter_fft(impulse, self.sigma)

        # Generate a 3D Gaussian kernel in the spatial domain
        coords = [torch.arange(s, device=self.device) - (s - 1) / 2 for s in self.shape]
        z, y, x = torch.meshgrid(*coords, indexing='ij')
        squared_dist = x ** 2 + y ** 2 + z ** 2

        # Below is the comparison of the filter with the analytical multi-normal distribution in 3 dimensions.
        scaling = 1.0 / (2 * math.sqrt(2) * math.sqrt(torch.pi ** 3) * self.sigma ** 3)
        gaussian = scaling * torch.exp(-squared_dist / (2 * self.sigma ** 2))
        self.assertTrue(torch.allclose(filtered, gaussian, atol=1e-4))

        # The same is true if we just take a Gaussian function with an appropriate sigma and normalize it.
        gaussian = torch.exp(-squared_dist / (2 * self.sigma ** 2))
        gaussian /= torch.sum(gaussian)
        self.assertTrue(torch.allclose(filtered, gaussian, atol=1e-4))


    def test_energy_reduction(self):
        """
        Energy Reduction Test: Confirm that the filter reduces the energy of the input signal,
        which a blurring filter should do.
        """
        input_tensor = torch.randn(self.shape, device=self.device)
        input_energy = torch.sum(input_tensor**2).item()
        filtered = gaussian_filter_fft(input_tensor, self.sigma)
        output_energy = torch.sum(filtered**2).item()
        self.assertTrue(input_energy/output_energy > 300.0)

    def test_linearity(self):
        """
        Linearity Test: Verify the linearity property of the filter.
        Assert that F(a + b) == F(a) + F(b).
        """
        a = torch.randn(self.shape, device=self.device)
        b = torch.randn(self.shape, device=self.device)

        filtered_sum = gaussian_filter_fft(a + b, self.sigma)

        filtered_A = gaussian_filter_fft(a, self.sigma)
        filtered_B = gaussian_filter_fft(b, self.sigma)
        sum_filtered = filtered_A + filtered_B

        self.assertTrue(torch.allclose(filtered_sum, sum_filtered, atol=1e-4))

    def test_shift_invariance(self):
        """
        Shift Invariance Test: Ensure the filter's response is shift-invariant.
        """
        input_tensor = torch.randn(self.shape, device=self.device)

        shift = 5
        shifted_tensor = torch.roll(input_tensor, shifts=shift, dims=2)

        filtered_original = gaussian_filter_fft(input_tensor, self.sigma)
        filtered_shifted = gaussian_filter_fft(shifted_tensor, self.sigma)

        rolled_filtered = torch.roll(filtered_original, shifts=shift, dims=2)
        self.assertTrue(torch.allclose(filtered_shifted, rolled_filtered, atol=1e-4))

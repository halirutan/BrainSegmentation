# Brain Segmentation

Welcome to the Brain Segmentation project documentation.

## Overview

This project provides tools and algorithms for brain segmentation tasks.
Brain segmentation is the process of identifying and isolating different structures within brain images,
which is crucial for various medical and research applications.
Here, we aim to improve segmentation results for resolutions higher than 1 mm and to better handle typical bias
artifacts we see in 7T scans.

To achieve that, we will create synthetic training datasets tailored to the resolution and contrasts we have.
We plan to train different neural network model architectures and verify their performance against the
[UltraCortex](https://www.ultracortex.org/) dataset's manual segmentations.
We hope that by training specific contrasts instead of being contrast-agnostic,
the models can learn unique features of our images and segmentation on high-contrast images can be improved 
significantly.

The approach consists of the following steps:

- Downloading large datasets and create (good-enough) full-head segmentations that serve as _label training maps_.
  Training label maps are used to create synthetic training pairs that contain various forms of augmentation and use
  a sampling model that creates "fake" MRI images that fit the segmentation labels perfectly.
- Creating the actual training data includes upsampling the training label maps to the target resolution.
  Additionally, a statistical contrast analysis of the MRI images that we want to segment is performed, which results
  in detailed information about how to sample different regions of the synthetic MRI images. With this, it becomes
  possible to generate arbitrary many randomly sampled and randomly augmented training pairs.
- Selecting and training a neural network model for high-resolution data brings its own set of challenges, such as
  dealing with memory constraints on today's GPUs. We will consider different strategies from different angles like
  patchwise or sliding window training, getting hold of NVidia GPUs larger than the 48GB available, or training on
  AMD APUs which have shared CPU/GPU memory.
- Once the training is done, we will verify the quality of the segmentation with different measures against the
  manual segmentations provided in the UltraCortex dataset. 


## Getting Started

- [Installation](installation.md)
- [Contributing](contribute/contributing.md)

## Features

- Scale label images for brain segmentation
- More features coming soon...

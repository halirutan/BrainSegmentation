# Charm Segmentation

This document provides a detailed overview of the CHARM (SimNIBS) segmentation installation, execution, and setup.

---

## Description of the CHARM Method

The SimNIBS package is usually used to calculate electric fields caused by Transcranial Electrical Stimulation (TES) and
Transcranial Magnetic Stimulation (TMS).
For this project, we will only use the full head segmentation functionality of SimNIBS to get
labeled images for the regions outside the brain.
Later we will merge these outer parts with the brain segmentation from a different, high-quality algorithm.

### Pipeline Steps

The complete SimNIBS pipeline is divided into three parts:

1. Automatic segmentation of MRI images and meshing to create individualized head models
2. Calculation of electric fields through the Finite Element Method (FEM)
3. Post-processing of results for further analysis.

For this project, we will only use the first step.

---

## Environment Preparation

### Prerequisites

Ensure that the following software is present:

- Python 3.8 or newer
- Installed miniforge

### Environment Installation

The first step is to clone the latest release of the SimNIBS repository from GitHub:

```shell
git clone --depth 1 https://github.com/simnibs/simnibs.git --branch v4.5.0
```

To install SimNIBS into a new environment, use the following steps:

```shell
cd simnibs
source conda_path/etc/profile.d/conda.sh
source conda_path/etc/profile.d/mamba.sh

mamba env create -f environment_linux.yml # we assume a Linux environment here!
conda activate simnibs_env
pip install -E .
python simnibs/cli/link_external_progs.py

# If you need fsleyes for viewing, uncomment the next line
# pip install fsleyes
```

## Running the Segmentation

After installing SimNIBS, you should verify that it works correctly.
To do this, use an example Nifti file with `.nii` extension and execute the following command:

```
charm "file_id" image/path/example.nii
```

where

1. `file_id` is a name chosen by the user for this segmentation (it is recommended to use the same name as the MRI file that needs to be segmented).
2. `image/path/example.nii` is the path to the MRI file that needs to be segmented.

### Inputs of Segmentation

`example.nii` - a file with the MRI image that needs to be segmented.

### Outputs of Segmentation

- `m2m_file_id` - A directory containing all the information about segmentation and electrode connections.
- `labeling.nii.gz` - A specific file required for further work, located in the `m2m_file_id` directory at the 
   path `image/path/m2m_file_id/segmentation/labeling.nii.gz`

### Example

The following Python code demonstrates how to segment all MRI images in a specified directory:

```shell
import os
import time

#Enter into the variables the path to a directory with MRI files
path = "/example/path/Try_A/Dir_Seg"
path_res = path + "/../Dir_Seg_Res"
os.system("mkdir " + path_res)
for file in os.listdir(path):
    time.sleep(1)
    name = os.path.basename(file).split('.')[0]
    com = "touch " + path + "/Process.txt; charm " + name + " " + os.path.abspath(file) + "; mv " + path + "/m2m_" + name + " " + path_res + "; rm -f " + path + "/Process.txt"
    os.system(com)
    time.sleep(1)
    while os.path.exists(path + "/Process.txt"):
        time.sleep(5)
```

To run the code:

1. Enter the path to the directory containing the MRI files into the "path" variable in the code.
2. Run the "python" command in the SimNIBS environment in the Terminal.

---

## Viewing Results

After the segmentation is complete, you can view the results using the FSLEyes program. For the directory that contains the results of the MRI image segmentation, enter the following command:

`fsleyes #PATH/m2m_"Name of File"/segmentation/labeling.nii.gz`

This launches the FSLEyes program to view the segmentation results, where #PATH is the path to the directory with the results.

### Interface of FSLEyes

In the opened window, you can see the results and adjust the settings as needed. To clearly view the organs from the CHARM segmentation, use the following settings in the top of the window:

- In "Labeling", select "Label Image" to see each organ with its particular label.

- Set "Outline width" to be thick enough so that the boundary lines are clearly visible without blocking any details.

- In "Look-up Table", use Random (big) or FreeSurferColorLut to see all the head segmentation with all organs clearly and brightly.

- The rest of the settings: "Brightness", "Contrast", "Zoom", "Opacity", can be adjusted according to your preference, but these values are recommended:
  - Brightness: Middle
  - Contrast: Middle
  - Zoom: 100
  - Opacity: Maximum (the right end)

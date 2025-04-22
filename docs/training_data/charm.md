# Charm Segmentation First Run

In this document is described a detailed overview of the CHARM (SimNIBS) segmentation installation, setting up and running.

---

## Description of the CHARM Method

The main goal of SimNIBS is to calculate electric fields caused by Transcranial Electrical Stimulation (TES) and Transcranial Magnetic Stimulation (TMS).

### Pipeline Steps

The pipeline is divided in three parts:

1. Automatic segmentation of MRI images and meshing to create individualized head models
2. Calculation of electric fields through the Finite Element Method (FEM)
3. Post-processing of results for further analysis.

---

## Environment's Preparation

### Prerequisites

Ensure that the following software and environments were set up:

-Python 3.8 or newer

-Installed Conda

-Installed file environment_linux.yml (From website https://github.com/simnibs/simnibs/releases/tag/v4.5.0)

### Environment Installation

Was chosen data where conda had to install new environment and chose way in which this environment can be installed (free conda). The process of installation was contained these commands: 

-`source #PATH/conda.sh`: Referencing a conda file so that commands associated with it will work, where #PATH is path to this file in file system.

-`conda env create -f #PATH/environment_linux.yml`: Creates Conda environment for SimNIBS, where #PATH is path to installation file in file system.

-`conda activate simnibs_env`: Activated Created in previous command Cinda environment.

-`pip install -f https://github.com/simnibs/simnibs/releases/latest simnibs`: Installs in created Conda environment needed files from GitHub.

---

## Running the Segmentation

Now SimNIBS is installed and should be checked if it works. For this was used one example file resolution “nii” and executed command: 

`charm “File ID” #PATH/”Name of File”.nii`: 

1. "File ID" is a chosen by user name for this segmentation (recomended to use the same name as was used for MRI which has to be segmented).

2. #PATH is a path to the file with MRI which is needed to segment.

### Inputs of Segmentation

File ID.nii - file with MRI image which has to be segmented.

### Outputs of Segmentation

m2m_File ID - Directory with all the information about segmentation and electrodes connection. it is needed only file ... which located ...

... - a specific file required for further work which is located in m2m_File ID directory along the path ...

### Example

As an example of using segmentation it was written code in Python during the execution of which all the MRI images in separated directory are segmented:

```shell
import os
import time

#Enter into the variables the path to a directory with MRI files
path = "/data/pt_np-pscheibe/datasets/test_data/Try_A/Dir_Seg"
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

1. Must be entered path to directory with needed MRIes into variable "path" in the code.

2. Must be launched command "Python" in launched created environment SimNIBS in Terminal.

---

## Results Setting Up
### Interface of SLReyes

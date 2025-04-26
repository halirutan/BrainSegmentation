# Charm Segmentation First Run

In this document is described a detailed overview of the CHARM (SimNIBS) segmentation installation, running and setting up.

---

## Description of the CHARM Method

The main goal of SimNIBS is building of the full head segmentation from given MRI image and  calculating electric fields caused by Transcranial Electrical Stimulation (TES) and Transcranial Magnetic Stimulation (TMS). For this project is needed only function of making the full head segmentation, however, since the Brain segmentation is not so useful in this segmentation later it will be replaced by Synth_Seg segmentation.

### Pipeline Steps

The pipeline is divided in three parts (for this project is needed only the first one):

1. Automatic segmentation of MRI images and meshing to create individualized head models
2. Calculation of electric fields through the Finite Element Method (FEM)
3. Post-processing of results for further analysis.

---

## Environment's Preparation

### Prerequisites

Ensure that the following software and environments were set up:

-Python 3.8 or newer

-Installed Conda

-Installed FSLEyes program

-Installed file environment_linux.yml (From website https://github.com/simnibs/simnibs/releases/tag/v4.5.0)

### Environment Installation

Must be chosen data where Conda have to install new environment and must be chosen way in which this environment can be installed (free Conda). The process of installation was contained these commands: 

- `source #PATH/conda.sh`: Referencing a Conda file so that commands associated with it will work, where #PATH is path to this file in file system.

-`conda env create -f #PATH/environment_linux.yml`: Creates Conda environment for SimNIBS, where #PATH is path to installation file in file system.

-`conda activate simnibs_env`: Activates created in previous command Conda environment.

-`pip install -f https://github.com/simnibs/simnibs/releases/latest simnibs`: Installs in created Conda environment needed files from GitHub.

---

## Running the Segmentation

Now SimNIBS is installed and should be checked if it works. For this must be used one example file resolution “nii” and executed command: 

`charm “File ID” #PATH/”Name of File”.nii`: 

1. "File ID" is a chosen by user name for this segmentation (recomended to use the same name as was used for MRI which has to be segmented).

2. #PATH is a path to the file with MRI which is needed to segment.

### Inputs of Segmentation

File ID.nii - file with MRI image which has to be segmented.

### Outputs of Segmentation

m2m_File ID - Directory with all the information about segmentation and electrodes connection. 

labeling.nii.gz - a specific file required for further work which is located in m2m_File ID directory along the path #PATH/m2m_File ID/segmentation/ labeling.nii.gz, where #PATH is a path to the resulted directory with segmented MRI.

### Example

As an example of using of the segmentation it was written code in Python during the execution of which all the MRI images in separated directory are segmented:

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

2. Must be launched command "python" in launched created environment SimNIBS in Terminal.

---

## Results Setting Up

Now the needed segmentation was built and it is needed to open it. To see it will be used FSLEyes program. For directory which has got as a result of MPI image segmentation should be entered the command:

`fsleyes #PATH/m2m_”Name of File”/segmentation/labeling.nii.gz`: Launching the program FSLEyes to see the result of made segmentation, where #PATH is path to the directory with results.

### Interface of SLReyes

In opened window can be seen the result and it is ready for setting up and get the needed image. To check the organs which are needed from CHARM segmentation clearly are needed such a settings in the top of the window: 

-In “Labeling” must be chosen “Label Image” to see every organ with it’s particular label; 

-“Outline width” must be made as thick as it was possible so that the line doesn't block anything and at the same time the boundary lines can be clearly seen; 

-In “Look-up Table” must be used Random (big) or FreeSurferColorLut to see clearly and brightly all the head segmentation with all the organs;

-The rest of the settings: “Brightness”, “Contrast”, “Zoom”, “Opacity”, can be chosen as it is comfortable for user, but these positions and amounts are recommended: 

	-Brightness: Middle;
-Contrast: Middle;
-Zoom: 100;
-Opacity: The right end.

---


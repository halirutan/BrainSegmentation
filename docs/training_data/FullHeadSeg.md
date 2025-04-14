## FullHead Segmentation Pipeline Documentation

This document provides a detailed overview of the FullHead segmentation pipeline, combining SynthSeg and CHARM segmentations from brain MRI datasets to create consistent overlay segmentation maps.

# Overview

The pipeline integrates two advanced segmentation methods:

SynthSeg: A robust segmentation method using machine learning, particularly suited for clinical datasets.

CHARM (SimNIBS): A segmentation tool primarily used for head modeling in neurostimulation applications.

The combination ensures precise and consistent segmentation, suitable for generating training data or subsequent analyses.

## Usage

# Prerequisites

Ensure the following software and environments are set up:

Python 3.8 or newer

Conda environments:

synth_seg_orig_py38

simnibs_env

SynthSeg script and directory paths:

Path to SynthSeg_predict.py script

Path to SynthSeg directory

# Running the Pipeline

The pipeline is executed via command line:

python create_fh_seg.py <input_dir> <output_dir> <synthseg_script> <synthseg_dir>

# Arguments:

<input_dir>: Directory containing input NIfTI files (.nii or .nii.gz).

<output_dir>: Directory where the segmentation overlays will be stored.

<synthseg_script>: Absolute path to SynthSeg_predict.py.

<synthseg_dir>: Absolute path to the SynthSeg root directory.

# Example

python create_fh_seg.py \
    /data/your_project/input_niftis \
    /data/your_project/output_overlays \
    /path/to/SynthSeg/scripts/commands/SynthSeg_predict.py \
    /path/to/SynthSeg

# Output

The pipeline performs the following steps:

SynthSeg segmentation: Automatically segments brain structures robustly from the input MRI.

CHARM segmentation: Performs head model segmentation, including specific anatomical labels.

Label isolation and resampling: SynthSeg segmentation is resampled to match CHARM segmentation resolution.

Overlay creation: Combines both segmentations into a unified segmentation map, prioritizing SynthSeg labels.

Label consistency verification: Ensures all generated overlays share identical sets of segmentation labels.

# Output Files

Output overlays are stored with filenames structured as:

<basename_of_input_file>_overlay.nii.gz

#Label Consistency Check

The pipeline automatically verifies label consistency across all generated segmentation overlays. If any discrepancies occur, the process halts and clearly identifies the inconsistent files.

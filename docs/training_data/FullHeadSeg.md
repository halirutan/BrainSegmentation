## FullHead Segmentation Pipeline Documentation

This document provides a detailed overview of the FullHead segmentation pipeline,
combining SynthSeg and CHARM segmentations from brain MRI datasets to create consistent overlay segmentation maps.

---

## Overview

The pipeline integrates two advanced segmentation methods:

- **SynthSeg:** A robust segmentation method using machine learning, particularly suited for clinical datasets.
- **CHARM (SimNIBS):** A segmentation tool primarily used for head modeling in neurostimulation applications.

The combination ensures precise and consistent segmentation, suitable for generating training data or subsequent analyses.

---

## Usage

### Prerequisites

Ensure the following software and environments are set up:

- **Python 3.8 or newer**
- **Conda environments:**
  - `synth_seg_orig_py38`
  - `simnibs_env`
  - For FullHead segmentation, ensure the availability of both SynthSeg and CHARM environments.

- **SynthSeg script and directory paths:**
  - Path to `SynthSeg_predict.py` script
  - Path to SynthSeg root directory

---

### Running the Pipeline

Execute the pipeline via the command line:

```shell
python create_fh_seg.py \
  --input_dir <input_dir> \
  --output_dir <output_dir> \
  --synthseg_script <synthseg_script> \
  --synthseg_dir <synthseg_dir> \
  --synthseg_env <synthseg_env> \
  --charm_env <charm_env>
```

### Arguments:

- `--input_dir`:  
  Path to the directory containing input NIfTI files (`.nii` or `.nii.gz`).

- `--output_dir`:  
  Path to the directory where the generated segmentation overlays will be saved.

- `--synthseg_script`:  
  Absolute path to the `SynthSeg_predict.py` script.

- `--synthseg_dir`:  
  Absolute path to the SynthSeg root directory.

- `--synthseg_env`:  
  Name of the conda environment used for running SynthSeg.

- `--charm_env`:  
  Name of the conda environment used for running CHARM (SimNIBS).

### Example

```shell
python create_fh_seg.py \
  --input_dir /path/to/input_dir \
  --output_dir /path/to/output_dir \
  --synthseg_script /path/to/SynthSeg_predict.py \
  --synthseg_dir /path/to/SynthSeg \
  --synthseg_env synth_seg_orig_py38 \
  --charm_env simnibs_env
```

---

## Pipeline Steps

The pipeline performs the following sequential steps:

1. **SynthSeg Segmentation:** Automatically segments brain structures robustly from the input MRI.
2. **CHARM Segmentation:** Performs head model segmentation, including specific anatomical labels.
3. **Label Isolation and Resampling:** Specific CHARM labels (501, 502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 520, 530) are isolated, and SynthSeg segmentation is resampled to match the CHARM segmentation resolution.
4. **Overlay Creation:** Combines both segmentations into a unified segmentation map, with SynthSeg labels taking precedence over CHARM labels where they overlap (SynthSeg labels are used wherever they exist, and CHARM labels are used only in areas where SynthSeg does not provide a label).
5. **Label Consistency Verification:** Ensures all generated overlays share identical sets of segmentation labels.

---

## Output Files

### Final Output
The main output overlays are stored with filenames structured as:

```shell
<basename_of_input_file>_overlay.nii.gz
```

### Intermediate Files
The pipeline also generates several intermediate files during processing:

- `<basename_of_input_file>_synthseg.nii.gz`: SynthSeg segmentation output
- `<basename_of_input_file>_charm/`: Directory containing CHARM segmentation results
- `<basename_of_input_file>_charm_filtered.nii.gz`: CHARM segmentation with isolated labels
- `<basename_of_input_file>_synthseg_resampled.nii.gz`: SynthSeg segmentation resampled to match CHARM resolution

## Label Consistency Check

The pipeline automatically verifies label consistency across all generated segmentation overlays.
If discrepancies are found, the process halts and explicitly identifies the inconsistent files.

---

### Testing

Unit tests are provided to ensure functionality:

- **`test_isolate_labels`:** Confirms the isolation of specified labels within segmentation data.
- **`test_resample_to_target`:** Validates correct resampling of segmentations to target resolutions.
- **`test_verify_label_consistency_across_overlays`:** Checks consistency of segmentation labels across multiple overlay files.

Ensure tests are run in environments with appropriate SynthSeg and CHARM setups for complete pipeline validation.

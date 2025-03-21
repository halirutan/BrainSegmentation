## FullHead Segmentation Pipeline Documentation

This document provides a detailed overview of the FullHead segmentation pipeline, combining SynthSeg and CHARM segmentations from brain MRI datasets to create consistent overlay segmentation maps.

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
  - For FullHead segmentation, ensure availability of both SynthSeg and CHARM environments.

- **SynthSeg script and directory paths:**
  - Path to `SynthSeg_predict.py` script
  - Path to SynthSeg root directory

---

### Running the Pipeline

Execute the pipeline via the command line:

```bash
python create_fh_seg.py <input_dir> <output_dir> <synthseg_script> <synthseg_dir>
```

### Arguments:

- `<input_dir>`: Directory containing input NIfTI files (`.nii` or `.nii.gz`).
- `<output_dir>`: Directory where the segmentation overlays will be stored.
- `<synthseg_script>`: Absolute path to `SynthSeg_predict.py`.
- `<synthseg_dir>`: Absolute path to the SynthSeg root directory.

### Example

```bash
python create_fh_seg.py \
    /data/your_project/input_niftis \
    /data/your_project/output_overlays \
    /path/to/SynthSeg/scripts/commands/SynthSeg_predict.py \
    /path/to/SynthSeg
```

---

## Pipeline Steps

The pipeline performs the following sequential steps:

1. **SynthSeg Segmentation:** Automatically segments brain structures robustly from the input MRI.
2. **CHARM Segmentation:** Performs head model segmentation, including specific anatomical labels.
3. **Label Isolation and Resampling:** SynthSeg segmentation is resampled to match the CHARM segmentation resolution.
4. **Overlay Creation:** Combines both segmentations into a unified segmentation map, prioritizing SynthSeg labels.
5. **Label Consistency Verification:** Ensures all generated overlays share identical sets of segmentation labels.

---

## Output Files

Output overlays are stored with filenames structured as:

```
<basename_of_input_file>_overlay.nii.gz
```

### Label Consistency Check

The pipeline automatically verifies label consistency across all generated segmentation overlays. If discrepancies are found, the process halts and explicitly identifies the inconsistent files.

---

### Testing

Unit tests are provided to ensure functionality:

- **`test_isolate_labels`:** Confirms the isolation of specified labels within segmentation data.
- **`test_resample_to_target`:** Validates correct resampling of segmentations to target resolutions.
- **`test_verify_label_consistency_across_overlays`:** Checks consistency of segmentation labels across multiple overlay files.

Ensure tests are run in environments with appropriate SynthSeg and CHARM setups for complete pipeline validation.

# SynthSeg (Original) Segmentation

SynthSeg is a deep learning-based method for brain tumor segmentation.
We use the original implementation from [SynthSeg](https://github.com/BBillot/SynthSeg) to create the segmentation of
the brain that is subsequently combined with the outer head segmentation of [CHARM](./charm.md) to create the final
[full head segmentation](./FullHeadSeg.md)

## Building a Python Environment for SynthSeg

The package dependency files for SynthSeg can be found in the `python_setup/SynthSeg` directory.
This includes both the `environment.yml` file for conda/mamba environment setup and the `requirements.txt` file for pip dependencies.

First create a conda environment using miniforge and the `environment.yml` file:

```bash
# We use mamba here, but conda is also fine
cd python_setup/SynthSeg
mamba env create -f environment.yml
```

Activate the environment and use `pip` to install the rest of the dependencies:

```bash
conda activate synth_seg_orig_py38
pip install -r requirements.txt
```

This will set up a Python 3.8 environment with CUDA 10.1 support and all the necessary dependencies for running SynthSeg,
including TensorFlow GPU 2.2.0, Keras 2.3.1, and other required packages.

## Running SynthSeg

After setting up the Python environment, you can run SynthSeg using the `SynthSeg_predict.py` script. Below is a guide to the available options:

### Basic Usage

```bash
python ./scripts/commands/SynthSeg_predict.py --i <input_image_or_folder> --o <output_folder>
```

### Command-Line Options

| Option                   | Description                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| `--i I`                  | Input image(s) to segment. Can be a path to a single image or to a folder containing multiple images. |
| `--o O`                  | Output folder for segmentation results. Must be a folder if `--i` designates a folder.                |
| `--parc`                 | Perform cortex parcellation in addition to basic segmentation.                                        |
| `--robust`               | Use robust prediction mode for higher accuracy (slower processing).                                   |
| `--fast`                 | Bypass some postprocessing steps for faster predictions (may reduce accuracy).                        |
| `--ct`                   | Clip intensities to [0,80] range for CT scans.                                                        |
| `--vol VOL`              | Path to output CSV file with volumes (mm³) for all regions and subjects.                              |
| `--qc QC`                | Path to output CSV file with quality control scores for all subjects.                                 |
| `--post POST`            | Output folder for posterior probability maps. Must be a folder if `--i` designates a folder.          |
| `--resample RESAMPLE`    | Output folder for resampled images. Must be a folder if `--i` designates a folder.                    |
| `--crop CROP [CROP ...]` | Size of 3D patches to analyze. Default is 192.                                                        |
| `--threads THREADS`      | Number of CPU cores to use for processing. Default is 1.                                              |
| `--cpu`                  | Force CPU processing instead of GPU (useful when GPU memory is limited).                              |
| `--v1`                   | Use SynthSeg 1.0 (updated 25/06/22).                                                                  |

### Example Commands

Basic segmentation of a single image:
```bash
python ./scripts/commands/SynthSeg_predict.py --i input_brain.nii.gz --o output_folder/
```

Batch processing with robust mode and volume calculation:
```bash
python ./scripts/commands/SynthSeg_predict.py --i input_folder/ --o output_folder/ --robust --vol volume_stats.csv --threads 4
```

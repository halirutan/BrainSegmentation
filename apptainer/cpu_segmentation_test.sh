#!/bin/bash

source /opt/miniforge/etc/profile.d/conda.sh
conda activate synth_seg_orig_py38
cd /ptmp/pscheibe/models
python /opt/SynthSeg/scripts/commands/SynthSeg_predict.py --i example.nii --o out --robust --cpu --threads 144
conda deactivate


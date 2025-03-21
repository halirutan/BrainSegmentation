import os
import subprocess
import nibabel as nib
import numpy as np
import scipy.ndimage
import shutil
from simple_parsing import ArgumentParser
from dataclasses import dataclass

@dataclass
class CLIOptions:
    input_dir: str
    output_dir: str
    synthseg_script: str
    synthseg_dir: str
    synthseg_env: str
    charm_env: str

def run_synthseg(conda_env, input_nifti, output_nifti, synthseg_script, synthseg_dir):
    command = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {conda_env} && python {synthseg_script} --i {input_nifti} --o {output_nifti} --robust"
    subprocess.run(command, shell=True, check=True, executable="/bin/bash", cwd=synthseg_dir)

def run_charm(conda_env, subject_id, input_nifti, charm_working_dir):
    os.makedirs(charm_working_dir, exist_ok=True)
    subject_dir = os.path.join(charm_working_dir, f"m2m_{subject_id}")

    if os.path.exists(subject_dir):
        shutil.rmtree(subject_dir)

    command = (
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {conda_env} && "
        f"charm {subject_id} {input_nifti} --forcesform --forcerun"
    )

    subprocess.run(command, shell=True, check=True, executable="/bin/bash", cwd=charm_working_dir)

def isolate_labels(nifti_path, labels, output_nifti):
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.int32)
    isolated_data = np.where(np.isin(data, labels), data, 0)
    nib.save(nib.Nifti1Image(isolated_data, img.affine, img.header), output_nifti)

def resample_to_target(source_img, target_img):
    zoom_factors = np.array(target_img.shape) / np.array(source_img.shape)
    resampled_data = scipy.ndimage.zoom(source_img.get_fdata(), zoom_factors, order=0)
    return nib.Nifti1Image(resampled_data.astype(np.int32), target_img.affine, target_img.header)

def verify_label_consistency_across_overlays(overlay_paths):
    label_sets = [set(np.unique(nib.load(path).get_fdata().astype(np.int32))) - {0} for path in overlay_paths]
    common_labels = set.intersection(*label_sets)

    for i, labels in enumerate(label_sets):
        missing = labels.symmetric_difference(common_labels)
        if missing:
            raise ValueError(f"Inconsistent labels found in {overlay_paths[i]}: {missing}")

    print("Label consistency across all overlays successfully verified.")

def main(opts: CLIOptions):
    charm_labels = [501, 502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 520, 530]

    nifti_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(opts.input_dir)
        for file in files if file.endswith(('.nii', '.nii.gz'))
    ]

    print(f"Found NIfTI files: {len(nifti_files)}")

    overlay_paths = []

    for nifti_file in nifti_files:
        base_name = os.path.basename(nifti_file).replace('.nii.gz', '').replace('.nii', '')
        print(f"Processing {base_name}")

        synthseg_output = os.path.join(opts.output_dir, f"{base_name}_synthseg.nii.gz")
        charm_working_dir = os.path.join(opts.output_dir, f"{base_name}_charm")
        charm_filtered_path = os.path.join(opts.output_dir, f"{base_name}_charm_filtered.nii.gz")
        resampled_synthseg_output = os.path.join(opts.output_dir, f"{base_name}_synthseg_resampled.nii.gz")
        final_overlay = os.path.join(opts.output_dir, f"{base_name}_overlay.nii.gz")

        run_charm(opts.charm_env, base_name, nifti_file, charm_working_dir)
        run_synthseg(opts.synthseg_env, nifti_file, synthseg_output, opts.synthseg_script, opts.synthseg_dir)

        isolate_labels(
            os.path.join(charm_working_dir, f"m2m_{base_name}", "segmentation", "labeling.nii.gz"),
            charm_labels,
            charm_filtered_path
        )

        synthseg_img = nib.load(synthseg_output)
        charm_img = nib.load(charm_filtered_path)
        resampled_synthseg_img = resample_to_target(synthseg_img, charm_img)
        nib.save(resampled_synthseg_img, resampled_synthseg_output)

        charm_data = charm_img.get_fdata().astype(np.int32)
        synthseg_data = resampled_synthseg_img.get_fdata().astype(np.int32)

        overlay_data = np.where(synthseg_data > 0, synthseg_data, charm_data)
        nib.save(nib.Nifti1Image(overlay_data, charm_img.affine, charm_img.header), final_overlay)

        overlay_paths.append(final_overlay)
        print(f"Overlay saved: {final_overlay}")

    if len(overlay_paths) > 1:
        print("Verifying label consistency across generated overlays...")
        verify_label_consistency_across_overlays(overlay_paths)
    else:
        print("Only one overlay generated; consistency check not required.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CLIOptions, dest="opts")
    args = parser.parse_args()

    os.makedirs(args.opts.output_dir, exist_ok=True)
    main(args.opts)
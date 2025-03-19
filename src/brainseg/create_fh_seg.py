import os
import sys
import subprocess
import nibabel as nib
import numpy as np
import scipy.ndimage
import shutil

def run_synthseg(conda_env, input_nifti, output_nifti, synthseg_script, synthseg_dir):
    python_bin = os.path.join(conda_env, "bin", "python")
    env = os.environ.copy()
    env["PYTHONPATH"] = synthseg_dir

    subprocess.run(
        [python_bin, synthseg_script, "--i", input_nifti, "--o", output_nifti, "--robust"],
        check=True, cwd=synthseg_dir, env=env
    )

def run_charm(conda_env, subject_id, input_nifti, charm_working_dir):
    charm_bin = os.path.join(conda_env, "bin", "charm")
    subject_dir = os.path.join(charm_working_dir, f"m2m_{subject_id}")

    os.makedirs(charm_working_dir, exist_ok=True)

    if os.path.exists(subject_dir):
        shutil.rmtree(subject_dir)

    subprocess.run(
        [charm_bin, subject_id, input_nifti, "--forcesform", "--forcerun"],
        check=True, cwd=charm_working_dir
    )

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

def main(input_dir, output_dir, synthseg_script, synthseg_dir, synthseg_env, charm_env):
    charm_labels = [501, 502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 520, 530]

    nifti_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nifti_files.append(os.path.join(root, file))

    print(f"Found NIfTI files: {len(nifti_files)}")

    overlay_paths = []

    for nifti_file in nifti_files:
        base_name = os.path.basename(nifti_file).replace('.nii.gz', '').replace('.nii', '')
        print(f"Processing {base_name}")

        synthseg_output = os.path.join(output_dir, f"{base_name}_synthseg.nii.gz")
        charm_working_dir = os.path.join(output_dir, f"{base_name}_charm")
        charm_filtered_path = os.path.join(output_dir, f"{base_name}_charm_filtered.nii.gz")
        resampled_synthseg_output = os.path.join(output_dir, f"{base_name}_synthseg_resampled.nii.gz")
        final_overlay = os.path.join(output_dir, f"{base_name}_overlay.nii.gz")

        run_charm(charm_env, base_name, nifti_file, charm_working_dir)
        run_synthseg(synthseg_env, nifti_file, synthseg_output, synthseg_script, synthseg_dir)

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
    if len(sys.argv) != 7:
        print("Usage: python create_fh_seg.py <input_dir> <output_dir> <synthseg_script> <synthseg_dir> <synthseg_env> <charm_env>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    synthseg_script = sys.argv[3]
    synthseg_dir = sys.argv[4]
    synthseg_env = sys.argv[5]
    charm_env = sys.argv[6]

    os.makedirs(output_dir, exist_ok=True)

    main(input_dir, output_dir, synthseg_script, synthseg_dir, synthseg_env, charm_env)
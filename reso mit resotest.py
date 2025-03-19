import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from tqdm import tqdm


def gaussian_filter_mask(volume_3d: np.ndarray, label: float, sigma: float) -> np.ndarray:
    """
    Erzeugt eine Binärmaske (vol==label) und wendet Gaussian Filter an.
    Gibt ein float32-Array gleicher Größe zurück.
    """
    mask = (volume_3d == label).astype(np.float32)
    filtered = gaussian_filter(mask, sigma=sigma)
    return filtered.astype(np.float32)


def filter_and_argmax(volume_3d: np.ndarray, final_shape=(700,700,700), sigma: float=10.0) -> np.ndarray:
    """
    1) Skaliert 'volume_3d' per NN auf final_shape=(700,700,700).
    2) Gauss-Filter pro Label => Argmax => int32-Volume.
    """
    in_z, in_y, in_x = volume_3d.shape
    target_z, target_y, target_x = final_shape

    # Berechne individuellen Upscale-Faktor je Achse
    scale_z = target_z / in_z
    scale_y = target_y / in_y
    scale_x = target_x / in_x

    print(f"[INFO] Upscaling-Faktor = (z:{scale_z:.3f}, y:{scale_y:.3f}, x:{scale_x:.3f})")

    # NN-Upsample
    upscaled_3d = zoom(volume_3d, zoom=(scale_z, scale_y, scale_x), order=0)
    print(f"[INFO] Ergebnis-Volume Shape nach NN-Up = {upscaled_3d.shape}")

    # Argmax-Logik
    unique_labels = np.unique(upscaled_3d)
    print(f"[INFO] Labels im upgesampelten Bild: {unique_labels}")

    final_vol = np.zeros(upscaled_3d.shape, dtype=np.float32)
    max_gauss = np.zeros(upscaled_3d.shape, dtype=np.float32)

    for lab in tqdm(unique_labels, desc="Gauss-HPC"):
        gauss_val = gaussian_filter_mask(upscaled_3d, label=lab, sigma=sigma)
        update_mask = gauss_val > max_gauss
        final_vol[update_mask] = lab
        max_gauss[update_mask] = gauss_val[update_mask]

    return final_vol.astype(np.int32)


def main():

    image_file = "/Users/julietteburkhardt/SynthSeg/data/training_label_maps/training_seg_03.nii.gz"
    output_dir = "/Users/julietteburkhardt/BrainSegmentation/output"

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "output_700.nii.gz")

    # 1) Bild laden
    if not os.path.isfile(image_file):
        raise FileNotFoundError(f"Input nicht gefunden: {image_file}")
    nifti_in = nib.load(image_file)
    data_in = nifti_in.get_fdata().astype(np.int32)
    in_z, in_y, in_x = data_in.shape
    print(f"[INFO] Original-Volume Shape = {data_in.shape}")

    # 2) Input-Auflösung aus dem Header holen:
    input_zooms = nifti_in.header.get_zooms()[:3]
    print(f"[INFO] Input-Auflösung (mm/voxel) = {input_zooms}")

    # 3) HPC => 700³
    final_data = filter_and_argmax(data_in, final_shape=(700,700,700), sigma=10.0)

    # 4) Header/Affine anpassen
    final_header = nifti_in.header.copy()

    # Berechne, welcher Scale-Faktor je Achse nötig war
    scale_z = 700 / in_z
    scale_y = 700 / in_y
    scale_x = 700 / in_x

    # => Neue Auflösung = alteAuflösung / scale
    old_zooms = final_header.get_zooms()[:3]
    new_zooms = (
        old_zooms[0] / scale_z,
        old_zooms[1] / scale_y,
        old_zooms[2] / scale_x
    )
    final_header.set_zooms(new_zooms)

    print(f"[INFO] Output-Auflösung (mm/voxel) = {new_zooms}")

    final_affine = nifti_in.affine.copy()
    final_affine[0,0] /= scale_x
    final_affine[1,1] /= scale_y
    final_affine[2,2] /= scale_z

    final_header.set_qform(final_affine, code=1)
    final_header.set_sform(final_affine, code=1)

    # 5) Speichern
    nifti_out = nib.Nifti1Image(final_data, final_affine, final_header)
    nib.save(nifti_out, output_file)
    print(f"[DONE] Gespeichert: {output_file}")


if __name__ == "__main__":
    main()

import os
import dicom2nifti
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def convert_adni_dicom(adni_root, output_root):
    """
    Recursively finds and converts all MPRAGE DICOM files from the ADNI dataset to NIfTI format.
    :param adni_root: Path to the root ADNI directory
    :param output_root: Path to the output directory where NIfTI files will be saved
    """
    for patient in os.listdir(adni_root):
        patient_path = os.path.join(adni_root, patient)
        if not os.path.isdir(patient_path):
            continue

        mprage_path = os.path.join(patient_path, "MPRAGE")
        if not os.path.exists(mprage_path):
            continue

        for session in os.listdir(mprage_path):
            session_path = os.path.join(mprage_path, session)
            if not os.path.isdir(session_path):
                continue

            for scan_id in os.listdir(session_path):
                scan_path = os.path.join(session_path, scan_id)
                if not os.path.isdir(scan_path):
                    continue

                output_patient_folder = os.path.join(output_root, patient, session)
                os.makedirs(output_patient_folder, exist_ok=True)

                try:
                    print(f"Converting: {scan_path} -> {output_patient_folder}")
                    dicom2nifti.convert_directory(scan_path, output_patient_folder)
                except Exception as e:
                    print(f"Error converting {scan_path}: {e}")


def inspect_nifti(nifti_folder):
    """
    Loads and displays a slice of a NIfTI file for inspection.
    :param nifti_folder: Path to the folder containing NIfTI files.
    """
    for root, _, files in os.walk(nifti_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_path = os.path.join(root, file)
                img = nib.load(nifti_path)
                data = img.get_fdata()

                # Display the middle slice of the first axis
                slice_idx = data.shape[2] // 2
                plt.imshow(data[:, :, slice_idx], cmap="gray")
                plt.title(f"Slice {slice_idx} of {file}")
                plt.colorbar()
                plt.show()

                return  # Only show the first image for quick inspection


if __name__ == "__main__":
    adni_root = "./ADNI"  # Change to the actual path of your ADNI folder
    output_root = "./ADNI_NIfTI"  # Output directory for converted NIfTI files

    # convert_adni_dicom(adni_root, output_root)
    # print("Conversion completed!")

    # Inspect the converted NIfTI files
    inspect_nifti(output_root)

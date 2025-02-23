import os
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_synthstrip(freesurfer_home, input_path, ss_output_path):
    """Runs SynthStrip on a single NIfTI file."""
    if os.path.exists(ss_output_path):  # Avoid redundant processing
        print(f"Skipping {input_path}, output already exists.")
        return

    try:
        env = os.environ.copy()
        env["FREESURFER_HOME"] = freesurfer_home
        env["SUBJECTS_DIR"] = os.path.join(freesurfer_home, "subjects")

        command = [
            "/bin/bash",
            "-c",  # Use bash explicitly
            f"source {freesurfer_home}/SetUpFreeSurfer.sh && "
            f"mri_synthstrip -i {input_path} -o {ss_output_path}",
        ]

        start_time = time.time()
        subprocess.run(command, check=True, env=env)
        elapsed_time = time.time() - start_time

        print(
            f"✔ Processed: {input_path} -> {ss_output_path} (Time: {elapsed_time:.2f}s)"
        )

        # Remove original unstripped file
        os.remove(input_path)

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to process {input_path}: {e}")


def skull_strip_nifti(base_dir, freesurfer_home="/Applications/freesurfer/7.4.1"):
    """Runs SynthStrip on NIfTI files in parallel while preserving folder structure."""
    tasks = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nii.gz") and not file.startswith("ss_"):
                input_path = os.path.join(root, file)
                ss_output_path = os.path.join(root, "ss_" + file)
                tasks.append((freesurfer_home, input_path, ss_output_path))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("✅ No new NIfTI files to process.")
        return

    print(f"🔍 Found {total_tasks} files to process.")

    start_time = time.time()

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_synthstrip, *task) for task in tasks]

        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()  # This will raise any exception caught in run_synthstrip
            except Exception as e:
                print(f"⚠️ Error processing task {i+1}: {e}")

    elapsed_time = time.time() - start_time
    print(f"✅ Finished processing all files in {elapsed_time:.2f}s.")


if __name__ == "__main__":
    DATA = "./ADNI_NII_stripping"

    # Call the function with your base directory
    skull_strip_nifti(DATA)

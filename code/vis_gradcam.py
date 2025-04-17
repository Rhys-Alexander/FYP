import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from math import ceil


def collate_gradcam_visualizations_matplotlib(
    input_dir,
    output_path="gradcam_collage_matplotlib.png",
    categories=["AD_correct", "CN_correct", "AD_incorrect", "CN_incorrect"],
    slices=[55, 65, 75],
    subjects_per_category=3,
    label_font_size=10,
    title_font_size=12,
    dpi=150,
):
    """
    Collates individual Grad-CAM slice images into a single grid visualization
    using Matplotlib.

    Args:
        input_dir (str): Path to the directory containing the Grad-CAM PNG images.
        output_path (str): Path to save the final collage image.
        categories (list): List of category strings (e.g., 'AD_correct').
        slices (list): List of slice indices used in the filenames.
        subjects_per_category (int): The number of subjects visualized per category.
        label_font_size (int): Font size for category labels.
        title_font_size (int): Font size for main title and column titles.
        dpi (int): Resolution for the saved figure.
    """
    print(f"Starting Matplotlib collage creation from images in: {input_dir}")
    print(f"Categories: {categories}")
    print(f"Slices: {slices}")
    print(f"Expected subjects per category: {subjects_per_category}")

    image_files = {}
    all_subject_ids = set()
    found_subjects_count = {}

    # --- Find image files and group by category and subject ---
    for cat in categories:
        parts = cat.split("_")
        true_label = 1 if parts[0] == "AD" else 0
        correct = parts[1] == "correct"
        pred_label = true_label if correct else 1 - true_label

        pattern = os.path.join(
            input_dir, f"*_true{true_label}_pred{pred_label}_slice*_gradcam.png"
        )
        matching_files = glob.glob(pattern)

        subjects_in_cat = {}
        for f in matching_files:
            basename = os.path.basename(f)
            try:
                subject_id = "_".join(basename.split("_")[:-4])
                if not subject_id:  # Handle cases where split might be unexpected
                    print(f"Warning: Could not extract subject ID from {basename}")
                    continue
                if subject_id not in subjects_in_cat:
                    subjects_in_cat[subject_id] = {}
                slice_num = int(basename.split("_")[-2].replace("slice", ""))
                if slice_num in slices:
                    subjects_in_cat[subject_id][slice_num] = f
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse filename {basename}: {e}")
                continue

        # Select the required number of subjects (sort for consistency)
        # Ensure subjects have all required slices if possible, otherwise just take first N
        valid_subjects = {
            subj: data
            for subj, data in subjects_in_cat.items()
            if len(data) == len(slices)  # Prioritize subjects with all slices
        }
        sorted_valid_subjects = sorted(list(valid_subjects.keys()))

        # If not enough subjects with all slices, add others
        other_subjects = sorted(
            [subj for subj in subjects_in_cat if subj not in valid_subjects]
        )
        combined_sorted_subjects = sorted_valid_subjects + other_subjects

        selected_subjects = combined_sorted_subjects[:subjects_per_category]

        image_files[cat] = {
            subj: subjects_in_cat[subj]
            for subj in selected_subjects
            if subj in subjects_in_cat
        }
        all_subject_ids.update(selected_subjects)
        found_subjects_count[cat] = len(selected_subjects)

        if len(selected_subjects) < subjects_per_category:
            print(
                f"Warning: Found only {len(selected_subjects)} subjects for category {cat}, expected {subjects_per_category}"
            )

    if not any(image_files.values()):
        print(
            "Error: No valid image files found matching the patterns. Check input_dir and filenames."
        )
        return

    # --- Determine Layout & Create Figure ---
    num_rows = len(categories)
    num_cols = subjects_per_category * len(slices)

    # Estimate figure size based on a sample image aspect ratio and desired DPI
    # Load one image to get dimensions
    img_w, img_h = 100, 100  # Default/fallback size
    try:
        first_cat = next(iter(image_files.keys()))
        first_subj = next(iter(image_files[first_cat].keys()))
        first_slice = next(iter(image_files[first_cat][first_subj].keys()))
        template_path = image_files[first_cat][first_subj][first_slice]
        with Image.open(template_path) as img_template:
            # Use original orientation dimensions for aspect ratio calculation
            img_w, img_h = img_template.size
    except Exception as e:
        print(
            f"Warning: Could not load template image to determine size, using default. Error: {e}"
        )

    # Calculate figure size (inches) - adjust multiplier as needed for spacing
    fig_width = num_cols * (img_w / dpi) * 1.5
    fig_height = num_rows * (img_h / dpi) * 1.5
    # Add extra height for titles/labels
    fig_height += 1.0  # Add an inch for top titles/spacing
    fig_width += 1.0  # Add an inch for side labels/spacing

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False, dpi=dpi
    )

    # --- Populate Grid ---
    for i, cat in enumerate(categories):
        subjects = list(
            image_files.get(cat, {}).keys()
        )  # Already sorted during selection
        num_found_subjects = found_subjects_count[cat]

        for subj_idx in range(subjects_per_category):
            if subj_idx < num_found_subjects:
                subj_id = subjects[subj_idx]
                subj_files = image_files[cat][subj_id]

                for slice_idx, slice_num in enumerate(slices):
                    col_idx = subj_idx * len(slices) + slice_idx
                    ax = axes[i, col_idx]
                    img_path = subj_files.get(slice_num)

                    if img_path and os.path.exists(img_path):
                        try:
                            img_data = mpimg.imread(img_path)
                            img_data = np.rot90(img_data, k=1)
                            ax.imshow(img_data)
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
                            ax.text(
                                0.5,
                                0.5,
                                "Error",
                                ha="center",
                                va="center",
                                fontsize=label_font_size,
                                color="red",
                                transform=ax.transAxes,
                            )
                    else:
                        # Handle missing image
                        ax.text(
                            0.5,
                            0.5,
                            "Missing",
                            ha="center",
                            va="center",
                            fontsize=label_font_size,
                            color="gray",
                            transform=ax.transAxes,
                        )
                        print(
                            f"Info: Image not found for {cat}, Subject {subj_idx+1} ({subj_id}), Slice {slice_num}"
                        )

                    ax.axis("off")  # Turn off axis lines and ticks

                    # --- Add Column Titles (Slice Number) ---
                    if i == 0:  # Only add to the top row
                        ax.set_title(
                            f"Slice {slice_num}", fontsize=label_font_size, pad=5
                        )

                    # --- Add Column Titles (Subject Number) ---
                    # Place above the first slice of each subject group in the top row
                    if i == 0 and slice_idx == 0:
                        # Use annotate for positioning above the slice title
                        axes[0, col_idx].annotate(
                            f"Subject {subj_idx+1}",
                            xy=(0.5, 1),
                            xytext=(0, 15),  # Adjust y offset
                            xycoords="axes fraction",
                            textcoords="offset points",
                            fontsize=title_font_size,
                            weight="bold",
                            ha="center",
                            va="baseline",
                        )
            else:
                # --- Handle missing subjects (draw empty axes) ---
                for slice_idx in range(len(slices)):
                    col_idx = subj_idx * len(slices) + slice_idx
                    ax = axes[i, col_idx]
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=label_font_size,
                        color="lightgray",
                        transform=ax.transAxes,
                    )
                    ax.axis("off")

        # --- Add Row Labels (Category) ---
        # Use annotate on the first axes of the row for better control
        label_text = cat.replace("_", " ").title()
        axes[i, 0].annotate(
            label_text,
            xy=(0, 0.5),
            xytext=(-10, 0),  # Adjust x offset for padding
            xycoords="axes fraction",
            textcoords="offset points",  # Position relative to axes
            fontsize=title_font_size,
            weight="bold",
            ha="right",
            va="center",
            rotation=90,
        )  # Vertical labels

    # --- Final Adjustments & Save ---
    fig.suptitle(
        "Grad-CAM Visualization Summary",
        fontsize=title_font_size + 2,
        weight="bold",
        y=0.98,
    )  # Adjust y position

    # Adjust layout to prevent labels/titles overlapping
    # Increase top margin for subject titles, left margin for category labels
    plt.subplots_adjust(
        left=0.1, right=0.98, top=0.88, bottom=0.02, wspace=0.1, hspace=0.2
    )

    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Matplotlib collage saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving Matplotlib collage image: {e}")

    plt.close(fig)  # Close the figure to free memory


# --- Example Usage ---
if __name__ == "__main__":
    # Import Image from PIL just for getting dimensions safely in main block
    from PIL import Image

    # --- Run the collation function ---
    input_image_directory = "./cam_visualizations"
    output_collage_file = "gradcam_summary_visualization_matplotlib.png"

    # Create dummy input directory and files for testing if they don't exist
    if not os.path.exists(input_image_directory):
        print(f"Creating dummy input directory: {input_image_directory}")
        os.makedirs(input_image_directory)
        # Create some dummy placeholder images (e.g., solid color)
        dummy_img = Image.new("RGB", (100, 120), color="skyblue")
        dummy_slices = [55, 65, 75]
        dummy_subjects = {
            "AD_correct": ["SubjA", "SubjB", "SubjC"],  # true=1, pred=1
            "CN_correct": ["SubjD", "SubjE", "SubjF"],  # true=0, pred=0
            "AD_incorrect": ["SubjG", "SubjH", "SubjI"],  # true=1, pred=0
            "CN_incorrect": ["SubjK", "SubjL", "SubjM"],  # true=0, pred=1
        }
        label_map = {"AD": 1, "CN": 0}
        pred_map = {"correct": True, "incorrect": False}

        for cat, subjects in dummy_subjects.items():
            parts = cat.split("_")
            true_label = label_map[parts[0]]
            correct = pred_map[parts[1]]
            pred_label = true_label if correct else 1 - true_label
            for subj in subjects:
                for slc in dummy_slices:
                    fname = f"{subj}_true{true_label}_pred{pred_label}_slice{slc}_gradcam.png"
                    fpath = os.path.join(input_image_directory, fname)
                    if not os.path.exists(fpath):
                        dummy_img.save(fpath)
        dummy_img.close()
        print("Dummy files created.")

    collate_gradcam_visualizations_matplotlib(
        input_dir=input_image_directory,
        output_path=output_collage_file,
        # Optional: Adjust parameters if needed
        categories=["AD_correct", "CN_correct", "AD_incorrect", "CN_incorrect"],
        slices=[55, 65, 75],
        subjects_per_category=3,
        dpi=150,
    )

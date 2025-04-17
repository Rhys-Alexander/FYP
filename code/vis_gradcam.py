import cv2
import numpy as np
import re
from collections import defaultdict
import os


def parse_filename(filename):
    """Parses the Grad-CAM filename to extract metadata."""
    match = re.match(r"(.+?)_true(\d)_pred(\d)_slice(\d+)_gradcam\.png", filename)
    if match:
        subject_id = match.group(1)
        true_label = int(match.group(2))
        pred_label = int(match.group(3))
        slice_idx = int(match.group(4))
        category = f"{'AD' if true_label == 1 else 'CN'}_{'correct' if true_label == pred_label else 'incorrect'}"
        return subject_id, category, slice_idx
    return None, None, None


def collate_gradcam_visualizations(
    image_dir,
    output_path="collated_gradcam.png",
    slices_to_include=[42, 64, 85],
    subjects_per_category=3,
    categories_ordered=["AD_correct", "CN_correct", "AD_incorrect", "CN_incorrect"],
    label_area_width=200,  # Space on the left for category labels
    background_color=(255, 255, 255),  # White background
):
    """
    Collates individual Grad-CAM slice images into a single grid visualization.

    Args:
        image_dir (str): Path to the directory containing the Grad-CAM PNG images.
        output_path (str): Path to save the final collated image.
        slices_to_include (list): List of slice indices to include, in desired order.
        subjects_per_category (int): Max number of subjects to display per category.
        categories_ordered (list): Order in which to display categories vertically.
        label_area_width (int): Width (in pixels) reserved for category labels on the left.
        background_color (tuple): RGB tuple for the background.
    """
    print(f"Scanning directory: {image_dir}")
    categorized_files = defaultdict(lambda: defaultdict(dict))
    all_files = [f for f in os.listdir(image_dir) if f.endswith("_gradcam.png")]

    if not all_files:
        print(f"Error: No '*_gradcam.png' images found in {image_dir}")
        return

    orig_img_h, orig_img_w = (
        0,
        0,
    )  # Dimensions of individual slice images BEFORE rotation

    # --- 1. Find and categorize images ---
    for filename in all_files:
        subject_id, category, slice_idx = parse_filename(filename)
        if subject_id and category and slice_idx in slices_to_include:
            filepath = os.path.join(image_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: File path does not exist: {filepath}")
                continue

            # Store filepath, keyed by category, then subject, then slice
            categorized_files[category][subject_id][slice_idx] = filepath

            # Get image dimensions from the first valid image found
            if orig_img_h == 0:
                try:
                    img = cv2.imread(filepath)
                    if img is None:
                        print(
                            f"Warning: Could not read image {filepath} to get dimensions."
                        )
                        continue
                    orig_img_h, orig_img_w, _ = img.shape
                    print(
                        f"Detected original image dimensions: {orig_img_w}x{orig_img_h}"
                    )
                except Exception as e:
                    print(f"Error reading image {filepath} for dimensions: {e}")
                    # Continue searching for a readable image

    if orig_img_h == 0 or orig_img_w == 0:
        print("Error: Could not determine image dimensions from any file.")
        return

    # --- 2. Define layout based on ROTATED dimensions ---
    img_h = orig_img_w  # Rotated height is original width
    img_w = orig_img_h  # Rotated width is original height
    print(f"Using rotated image dimensions for layout: {img_w}x{img_h}")

    num_categories = len(categories_ordered)
    num_slices = len(slices_to_include)

    # Calculate canvas size using rotated dimensions
    cell_w = img_w * num_slices  # Width for one subject (all slices horizontally)
    cell_h = img_h  # Height for one subject/category row
    grid_w = cell_w * subjects_per_category
    grid_h = cell_h * num_categories

    total_w = grid_w + label_area_width
    total_h = grid_h

    # Create blank canvas
    canvas = np.full((total_h, total_w, 3), background_color, dtype=np.uint8)
    print(f"Creating canvas of size: {total_w}x{total_h}")

    # --- 3. Populate the canvas ---
    print("Populating grid...")
    for cat_idx, category in enumerate(categories_ordered):
        if category not in categorized_files:
            print(f"Warning: No images found for category '{category}'. Skipping row.")
            continue

        subjects = list(categorized_files[category].keys())
        print(f"  Category '{category}': Found {len(subjects)} subjects.")

        subjects_to_display = sorted(subjects)[
            :subjects_per_category
        ]  # Sort for consistency

        for subj_idx, subject_id in enumerate(subjects_to_display):
            slice_files = categorized_files[category][subject_id]

            for slice_pos_idx, slice_idx in enumerate(slices_to_include):
                filepath = slice_files.get(slice_idx)  # Use .get() for safety

                # Calculate top-left corner for this slice image using rotated dimensions
                row_start = cat_idx * cell_h  # cell_h is based on rotated height
                col_start = (
                    label_area_width
                    + (subj_idx * cell_w)
                    + (
                        slice_pos_idx * img_w
                    )  # cell_w and img_w are based on rotated dimensions
                )

                if filepath and os.path.exists(filepath):
                    try:
                        img = cv2.imread(filepath)
                        if img is None:
                            raise ValueError("Image read returned None")

                        # --- ROTATE THE IMAGE ---
                        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                        # Ensure rotated image is correct size (optional, assumes consistency)
                        # Note: comparing with img_h (rotated height) and img_w (rotated width)
                        if (
                            rotated_img.shape[0] != img_h
                            or rotated_img.shape[1] != img_w
                        ):
                            print(
                                f"Warning: Resizing rotated image {filepath} from {rotated_img.shape[:2]} to {(img_h, img_w)}"
                            )
                            # cv2.resize takes (width, height)
                            rotated_img = cv2.resize(rotated_img, (img_w, img_h))

                        # Place ROTATED image on canvas using ROTATED dimensions
                        canvas[
                            row_start : row_start + img_h, col_start : col_start + img_w
                        ] = rotated_img
                    except Exception as e:
                        print(
                            f"Error processing {filepath}: {e}. Placing gray placeholder."
                        )
                        # Place placeholder using ROTATED dimensions
                        canvas[
                            row_start : row_start + img_h, col_start : col_start + img_w
                        ] = (
                            128,
                            128,
                            128,
                        )  # Gray placeholder
                else:
                    print(
                        f"Missing slice {slice_idx} for {subject_id} in {category}. Placing gray placeholder."
                    )
                    # Place placeholder using ROTATED dimensions
                    canvas[
                        row_start : row_start + img_h, col_start : col_start + img_w
                    ] = (
                        128,
                        128,
                        128,
                    )  # Gray placeholder

    # --- 4. Add category labels ---
    print("Adding labels...")
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 0.9
    label_thickness = 2
    label_color = (0, 0, 0)  # Black

    for cat_idx, category in enumerate(categories_ordered):
        # Calculate position for the label (centered vertically in the row, within the label area)
        # cell_h is now based on the rotated image height
        text = category.replace("_", " ")
        text_size = cv2.getTextSize(text, label_font, label_scale, label_thickness)[0]

        label_x = (
            label_area_width - text_size[0]
        ) // 2  # Center horizontally in label area
        label_y = (cat_idx * cell_h) + (
            cell_h + text_size[1]
        ) // 2  # Center vertically in the row (using updated cell_h)

        cv2.putText(
            canvas,
            text,
            (label_x, label_y),
            label_font,
            label_scale,
            label_color,
            label_thickness,
            cv2.LINE_AA,
        )

    # --- 5. Save the final image ---
    try:
        cv2.imwrite(output_path, canvas)
        print(f"Collated visualization successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving final image to {output_path}: {e}")


# --- How to use it ---
if __name__ == "__main__":
    # Directory where your individual Grad-CAM PNGs are stored
    grad_cam_image_directory = "./cam_visualizations"

    # Desired output file path
    collated_output_file = (
        "./collated_gradcam_visualization_rotated.png"  # Changed output name slightly
    )

    if not os.path.isdir(grad_cam_image_directory):
        print(
            f"Error: The specified image directory does not exist: {grad_cam_image_directory}"
        )
    else:
        collate_gradcam_visualizations(
            image_dir=grad_cam_image_directory, output_path=collated_output_file
        )

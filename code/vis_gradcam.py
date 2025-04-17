import os
import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math


def collate_gradcam_visualizations(
    input_dir,
    output_path="gradcam_collage.png",
    categories=["AD_correct", "CN_correct", "AD_incorrect", "CN_incorrect"],
    slices=[55, 65, 75],
    subjects_per_category=3,
    spacing=10,
    label_font_size=20,
):
    """
    Collates individual Grad-CAM slice images into a single grid visualization.

    Args:
        input_dir (str): Path to the directory containing the Grad-CAM PNG images.
        output_path (str): Path to save the final collage image.
        categories (list): List of category strings (e.g., 'AD_correct').
        slices (list): List of slice indices used in the filenames.
        subjects_per_category (int): The number of subjects visualized per category.
        spacing (int): Pixels between images and labels in the collage.
        label_font_size (int): Font size for category and slice labels.
    """
    print(f"Starting collage creation from images in: {input_dir}")
    print(f"Categories: {categories}")
    print(f"Slices: {slices}")
    print(f"Expected subjects per category: {subjects_per_category}")

    image_files = {}
    all_subject_ids = set()

    # Find image files and group by category and subject
    for cat in categories:
        # Extract label and prediction from category name
        parts = cat.split("_")
        true_label = 1 if parts[0] == "AD" else 0
        correct = parts[1] == "correct"
        pred_label = true_label if correct else 1 - true_label

        # Find subjects matching this category
        pattern = os.path.join(
            input_dir, f"*_true{true_label}_pred{pred_label}_slice*_gradcam.png"
        )
        matching_files = glob.glob(pattern)

        subjects_in_cat = {}
        for f in matching_files:
            basename = os.path.basename(f)
            try:
                # Extract subject ID (assuming format like 'ID_true..._gradcam.png')
                subject_id = "_".join(
                    basename.split("_")[:-4]
                )  # Adjust if ID format differs
                if subject_id not in subjects_in_cat:
                    subjects_in_cat[subject_id] = {}
                # Extract slice number
                slice_num = int(basename.split("_")[-2].replace("slice", ""))
                if slice_num in slices:
                    subjects_in_cat[subject_id][slice_num] = f
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse filename {basename}: {e}")
                continue

        # Select the required number of subjects (sort for consistency)
        selected_subjects = sorted(list(subjects_in_cat.keys()))[:subjects_per_category]
        image_files[cat] = {subj: subjects_in_cat[subj] for subj in selected_subjects}
        all_subject_ids.update(selected_subjects)

        if len(selected_subjects) < subjects_per_category:
            print(
                f"Warning: Found only {len(selected_subjects)} subjects for category {cat}, expected {subjects_per_category}"
            )
        elif len(selected_subjects) > subjects_per_category:
            # This case might happen if sorting changes which ones are picked by [:subjects_per_category]
            pass  # No need to print info if we just take the first N sorted ones

    if not any(image_files.values()):
        print(
            "Error: No valid image files found matching the patterns. Check input_dir and filenames."
        )
        return

    # --- Determine Layout ---
    num_rows_per_cat = subjects_per_category
    num_cols = len(slices)

    # Load one image to get dimensions BEFORE rotation
    first_cat_key = next(iter(image_files.keys()), None)
    if not first_cat_key or not image_files[first_cat_key]:
        print(
            "Error: No subjects found for the first category to determine image size."
        )
        return
    first_subj_key = next(iter(image_files[first_cat_key].keys()), None)
    if not first_subj_key or not image_files[first_cat_key][first_subj_key]:
        print(
            f"Error: No slices found for the first subject ({first_subj_key}) in category {first_cat_key}."
        )
        return
    first_slice_key = next(
        iter(image_files[first_cat_key][first_subj_key].keys()), None
    )
    if not first_slice_key:
        print(f"Error: Could not get first slice key for subject {first_subj_key}.")
        return

    try:
        template_path = image_files[first_cat_key][first_subj_key][first_slice_key]
        with Image.open(template_path) as img_template:
            orig_w, orig_h = img_template.size
        # Define layout dimensions based on ROTATED size
        img_w, img_h = orig_h, orig_w  # Swapped dimensions
        print(
            f"Detected original image size: {orig_w}x{orig_h}. Using rotated size for layout: {img_w}x{img_h}"
        )
    except Exception as e:
        print(f"Error loading template image '{template_path}': {e}")
        return

    # Calculate total dimensions using ROTATED image dimensions
    total_width = (num_cols * img_w) + ((num_cols + 1) * spacing)
    total_height = (
        (len(categories) * num_rows_per_cat * img_h)  # Height of all image rows
        + (
            (len(categories) * num_rows_per_cat) * spacing
        )  # Spacing below each image row
        + (
            len(categories) * (spacing + label_font_size)
        )  # Space for category labels + spacing above them
        + (
            spacing + label_font_size
        )  # Extra space for top slice labels + spacing above them
        + spacing  # Final spacing at the bottom
    )

    # Create canvas
    collage = Image.new("RGB", (total_width, total_height), color="white")
    draw = ImageDraw.Draw(collage)
    try:
        # Use a basic default font if specific one not found
        font = ImageFont.truetype("arial.ttf", label_font_size)
    except IOError:
        print("Arial font not found, using default PIL font.")
        font = ImageFont.load_default(size=label_font_size)  # Try specifying size

    # --- Paste Images and Add Labels ---
    current_y = spacing + label_font_size + spacing  # Start below top slice labels

    # Add Slice Labels at the top (centered above rotated image width)
    for j, slice_num in enumerate(slices):
        label_text = f"Slice {slice_num}"
        # Use textbbox for potentially more accurate size in newer Pillow versions
        try:
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:  # Fallback for older Pillow
            text_w, text_h = draw.textsize(label_text, font=font)

        # Center label within the column width (img_w is rotated width)
        label_x = spacing + (j * (img_w + spacing)) + (img_w // 2) - (text_w // 2)
        label_y = spacing
        draw.text((label_x, label_y), label_text, fill="black", font=font)

    for i, cat in enumerate(categories):
        # Add Category Label
        cat_label_text = cat.replace("_", " ").title()
        try:
            bbox = draw.textbbox((0, 0), cat_label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(cat_label_text, font=font)

        cat_label_x = spacing
        cat_label_y = current_y
        draw.text((cat_label_x, cat_label_y), cat_label_text, fill="black", font=font)
        current_y += label_font_size + spacing  # Move down past label and its spacing

        subjects = list(image_files.get(cat, {}).keys())
        for row_idx in range(subjects_per_category):
            current_x = spacing  # Reset X for each new row (subject)
            if row_idx < len(subjects):
                subj_id = subjects[row_idx]
                subj_files = image_files[cat][subj_id]

                for col_idx, slice_num in enumerate(slices):
                    img_path = subj_files.get(slice_num)
                    if img_path and os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            # --- ROTATE THE IMAGE ---
                            rotated_img = img.transpose(
                                Image.Transpose.ROTATE_270
                            )  # 90 degrees counter-clockwise
                            img.close()  # Close original image

                            # Check size against ROTATED dimensions (img_w, img_h)
                            if rotated_img.size != (img_w, img_h):
                                print(
                                    f"Warning: Resizing rotated image {os.path.basename(img_path)} from {rotated_img.size} to {(img_w, img_h)}"
                                )
                                rotated_img = rotated_img.resize(
                                    (img_w, img_h), Image.Resampling.LANCZOS
                                )

                            # Paste the ROTATED image
                            collage.paste(rotated_img, (current_x, current_y))
                            rotated_img.close()  # Close rotated image after pasting

                        except Exception as e:
                            print(f"Error processing/pasting image {img_path}: {e}")
                            # Draw a placeholder using ROTATED dimensions
                            draw.rectangle(
                                [
                                    current_x,
                                    current_y,
                                    current_x + img_w,  # Use rotated width
                                    current_y + img_h,  # Use rotated height
                                ],
                                outline="red",
                                width=2,
                            )
                            draw.text(
                                (current_x + 5, current_y + 5),
                                "Error",
                                fill="red",
                                font=font,
                            )
                    else:
                        # Draw placeholder if image missing using ROTATED dimensions
                        print(
                            f"Warning: Image not found for {cat}, Subject {row_idx+1} ({subj_id}), Slice {slice_num}"
                        )
                        draw.rectangle(
                            [
                                current_x,
                                current_y,
                                current_x + img_w,  # Use rotated width
                                current_y + img_h,  # Use rotated height
                            ],
                            outline="gray",
                            width=1,
                        )
                        draw.text(
                            (current_x + 5, current_y + 5),
                            "Missing",
                            fill="gray",
                            font=font,
                        )

                    current_x += (
                        img_w + spacing
                    )  # Move right by rotated width + spacing
            else:
                # Draw placeholders if fewer subjects than expected using ROTATED dimensions
                for col_idx, slice_num in enumerate(slices):
                    draw.rectangle(
                        [
                            current_x,
                            current_y,
                            current_x + img_w,
                            current_y + img_h,
                        ],  # Use rotated dimensions
                        outline="lightgray",
                        width=1,
                    )
                    draw.text(
                        (current_x + 5, current_y + 5),
                        f"N/A",
                        fill="lightgray",
                        font=font,
                    )
                    current_x += (
                        img_w + spacing
                    )  # Move right by rotated width + spacing

            current_y += (
                img_h + spacing
            )  # Move down by rotated height + spacing for next subject/row
        # No extra spacing needed here, spacing after last image row is handled by loop increment and total height calc

    # Save the final collage
    try:
        collage.save(output_path)
        print(f"Collage saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving collage image: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # --- Run the collation function ---
    input_image_directory = "./cam_visualizations"
    output_collage_file = "gradcam_summary_visualization_rotated.png"  # Updated name

    collate_gradcam_visualizations(
        input_dir=input_image_directory,
        output_path=output_collage_file,
        # Optional: Adjust parameters if needed
        # categories=['AD_correct', 'CN_correct'], # Example: only show correct ones
        # slices=[64], # Example: only show middle slice
        # subjects_per_category=2,
    )

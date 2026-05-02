import numpy as np
from tqdm import tqdm
from config import *
from sam3_segmenter import SAM3Segmenter
from utils import *

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_paths(root_dir):
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def preprocess_dataset():
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = list_image_paths(RAW_DATASET_DIR)
    segmenter = SAM3Segmenter()
    for img_path in tqdm(image_paths, desc="Processing OPTED"):
        image_pil = load_image(img_path)
        image_np = np.array(image_pil)
        final_mask = None
        for prompt in PROMPTS[::-1]:
            mask, score = segmenter.segment(image_pil, prompt)
            if mask is not None and score >= CONFIDENCE_THRESHOLD:
                final_mask = mask
                break
        if final_mask is None:
            continue
        masked = mask_background(image_np, final_mask)
        box = get_bounding_box(final_mask)
        if box is None:
            continue
        padded_box = add_padding(box, image_np.shape, PADDING_RATIO)
        cropped = crop_image(masked, padded_box)
        aligned = align_horizontal(cropped)
        resized = resize_lanczos(aligned, IMAGE_SIZE)
        relative_stem = img_path.relative_to(RAW_DATASET_DIR).with_suffix("")
        output_stem = "_".join(relative_stem.parts)
        mask_color = np.zeros((*final_mask.shape, 3), dtype=np.uint8)
        mask_color[final_mask > 0] = [186, 225, 250]  # light blue
        save_image(mask_color, MASK_DIR / f"{output_stem}_mask.png")
        save_image(aligned, CROPPED_DIR / f"{output_stem}_cropped.png")
        save_image(resized, RESIZED_DIR / f"{output_stem}_224.png")
if __name__ == "__main__":
    preprocess_dataset()

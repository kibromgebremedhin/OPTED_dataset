from pathlib import Path
RAW_DATASET_DIR = Path("raw_images")
OUTPUT_DIR = Path("outputs")
CROPPED_DIR = OUTPUT_DIR / "cropped_aligned"
RESIZED_DIR = OUTPUT_DIR / "resized_224"
MASK_DIR = OUTPUT_DIR / "masks"
VIS_DIR = OUTPUT_DIR / "visualizations"
IMAGE_SIZE = 224
PADDING_RATIO = 0.05
MASK_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.0
DEVICE = "cuda"
SAM_MODEL_ID = "CIDAS/clipseg-rd64-refined"
PRIMARY_PROMPT = "inner surface of eyelid with red tissue"
PROMPTS = [
    "red tissue inside eye",
    "inner surface of eyelid",
    "red lining inside eyelid",
    "membrane under eyelid",
    "inner surface of eyelid with red tissue"
]

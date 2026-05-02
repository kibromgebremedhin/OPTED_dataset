# OPTED: Open Preprocessed Trachoma Eye Dataset

## Overview

OPTED is the first publicly available, standardized trachoma eye image dataset created using **zero-shot SAM 3 segmentation**. The pipeline automatically detects and extracts the conjunctival region of interest from clinical photographs, producing analysis-ready images with consistent framing.

**Trachoma** is the world's leading infectious cause of blindness, caused by _Chlamydia trachomatis_. It remains endemic in over 40 countries, disproportionately affecting underserved communities. Early detection of trachoma signs — **TF** (trachomatous inflammation—follicular) and **TI** (trachomatous inflammation—intense) — is critical for timely treatment.

## Dataset Statistics

|           | Normal    | TF      | TI     | Total     |
| --------- | --------- | ------- | ------ | --------- |
| Train     | 1,741     | 227     | 15     | 1,983     |
| Val       | 373       | 49      | 3      | 425       |
| Test      | 373       | 48      | 3      | 424       |
| **Total** | **2,487** | **324** | **21** | **2,832** |

> After quality filtering (confidence ≥ 0.5), **2,236 images** are included.

## Directory Structure

```
OPTED_dataset/
├── metadata.csv                  # Per-image metadata
├── images_224/                   # Standardized 224×224 PNG images
│   ├── Normal/
│   ├── TF/
│   └── TI/
├── images_cropped/               # Cropped + aligned (original aspect ratio)
│   ├── Normal/
│   ├── TF/
│   └── TI/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### `images_224/` (Recommended)

Standardized 224×224 pixel PNG images. These are ready for direct use with standard CNN architectures (ResNet, EfficientNet, ViT, etc.) via `torchvision.datasets.ImageFolder`.

### `images_cropped/`

Cropped and aligned images at original aspect ratio. Use these when you want to apply custom resizing/augmentation.

### `metadata.csv`

| Column             | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `filename`         | Image filename (PNG)                                 |
| `label`            | Class label: `Normal`, `TF`, or `TI`                 |
| `split`            | Pre-defined split: `train`, `val`, or `test`         |
| `confidence_score` | SAM 3 segmentation confidence (0–1)                  |
| `original_width`   | Width of original clinical photograph (px)           |
| `original_height`  | Height of original clinical photograph (px)          |
| `crop_width`       | Width of the segmentation crop (px)                  |
| `crop_height`      | Height of the segmentation crop (px)                 |
| `mask_area_pct`    | Percentage of crop occupied by the segmentation mask |

### `splits/`

Pre-defined stratified train/val/test split files. Each line is `<class>/<filename>`.

## Quick Start (PyTorch)

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("OPTED_dataset/images_224", transform=transform)
```

To use the pre-defined splits:

```python
import pandas as pd
from torch.utils.data import Subset

meta = pd.read_csv("OPTED_dataset/metadata.csv")
train_files = set(meta[meta["split"] == "train"]["filename"])

indices = [i for i, (path, _) in enumerate(dataset.samples)
           if Path(path).name in train_files]
train_subset = Subset(dataset, indices)
```

## Processing Pipeline

1. **Input**: Clinical eyelid photographs (3008×2000 to 4288×2848 px)
2. **Segmentation**: SAM 3 (848M params) with text prompt _"inner surface of eyelid with red tissue"_
3. **Cropping**: Bounding-box extraction around the predicted mask
4. **Alignment**: Orientation normalization
5. **Standardization**: Resize to 224×224 px (bilinear interpolation)
6. **Quality filter**: Minimum confidence score ≥ 0.5

## Classes (WHO Simplified Grading)

- **Normal**: No signs of active trachoma
- **TF**: Trachomatous inflammation—follicular (≥5 follicles ≥0.5 mm in the central upper tarsal conjunctiva)
- **TI**: Trachomatous inflammation—intense (inflammatory thickening obscuring >50% of deep tarsal vessels)

## Important Notes

- **Class imbalance**: TI is severely underrepresented (21 images). Use appropriate sampling strategies (e.g., WeightedRandomSampler) or loss weighting.
- **Clinical context**: Labels reflect WHO simplified trachoma grading by trained graders.
- **Ethics**: This dataset was collected under appropriate ethical oversight. See the accompanying paper for details.

## Citation

If you use OPTED in your research, please cite:

```bibtex
@inproceedings{opted2025,
  title     = {OPTED: Open Preprocessed Trachoma Eye Dataset Using Zero-Shot SAM 3 Segmentation},
  author    = {[Authors]},
  booktitle = {[Conference]},
  year      = {2025}
}
```

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Preprocessed dataset

If you need the dataset, you can request for the whole dataset by sending an email to: **kibrom.gebremedhin@mu.edu.et**

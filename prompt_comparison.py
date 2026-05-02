import numpy as np
import matplotlib.pyplot as plt
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


def evaluate_prompts():
    image_paths = list_image_paths(RAW_DATASET_DIR)
    segmenter = SAM3Segmenter()
    results = {prompt: {"detections": 0, "misses": 0, "scores": [], "mask_areas": []} for prompt in PROMPTS}

    for img_path in tqdm(image_paths, desc="Evaluating prompts"):
        image_pil = load_image(img_path)
        image_np = np.array(image_pil)
        total_pixels = image_np.shape[0] * image_np.shape[1]

        for prompt in PROMPTS:
            mask, score = segmenter.segment(image_pil, prompt)
            if mask is not None and score >= CONFIDENCE_THRESHOLD:
                results[prompt]["detections"] += 1
                results[prompt]["scores"].append(score)
                results[prompt]["mask_areas"].append(np.sum(mask) / total_pixels)
            else:
                results[prompt]["misses"] += 1
                results[prompt]["scores"].append(0.0)
                results[prompt]["mask_areas"].append(0.0)

    metrics = {}
    for prompt in PROMPTS:
        r = results[prompt]
        total = r["detections"] + r["misses"]
        metrics[prompt] = {
            "Detection Rate": r["detections"] / total,
            "Score Mean": np.mean(r["scores"]),
            "Score Std": np.std(r["scores"]),
            "Mask Area Ratio": np.mean(r["mask_areas"]),
            "Miss (normalized)": r["misses"] / total,
        }

    print("\nPrompt Comparison Results:")
    print(f"{'Prompt':<45} {'Det Rate':<10} {'Miss':<6} {'Mean Score':<11} {'Std':<8} {'Area Ratio':<11}")
    print("-" * 100)
    for prompt in PROMPTS:
        m = metrics[prompt]
        print(f"{prompt:<45} {m['Detection Rate']:<10.4f} {results[prompt]['misses']:<6} {m['Score Mean']:<11.4f} {m['Score Std']:<8.4f} {m['Mask Area Ratio']:<11.4f}")

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = ["Detection Rate", "Score Mean", "Score Std", "Mask Area Ratio", "Miss (normalized)"]
    num_metrics = len(metric_names)
    num_prompts = len(PROMPTS)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    for i, prompt in enumerate(PROMPTS):
        values = [metrics[prompt][m] for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=prompt, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ["Detection Rate", "Score Mean", "Score Std", "Mask Area Ratio", "Miss"],
        fontsize=10,
    )
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Prompt Comparison - Radar Chart", pad=20, fontsize=14)
    plt.tight_layout()

    output_path = VIS_DIR / "prompt_comparison_radar.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved radar chart: {output_path}")


if __name__ == "__main__":
    evaluate_prompts()

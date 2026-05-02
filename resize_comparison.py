import cv2
import matplotlib.pyplot as plt
import pandas as pd
from config import CROPPED_DIR, VIS_DIR
from metrics import compute_metrics


def compare_interpolations(image):
    methods = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    results = {}
    for name, method in methods.items():
        resized = cv2.resize(image, (224,224), interpolation=method)
        psnr_score, ssim_score = compute_metrics(image, resized)
        results[name] = {"psnr": psnr_score, "ssim": ssim_score}
    return results


def run_comparison():
    image_paths = sorted(CROPPED_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No cropped images found in {CROPPED_DIR}")

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metrics_by_method = compare_interpolations(image)
        for method, scores in metrics_by_method.items():
            rows.append(
                {
                    "image": image_path.name,
                    "method": method,
                    "psnr": scores["psnr"],
                    "ssim": scores["ssim"],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid images were processed for interpolation comparison.")

    summary = df.groupby("method", as_index=False)[["psnr", "ssim"]].mean()
    summary = summary.sort_values(by="psnr", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(summary["method"], summary["psnr"], color="#4C78A8")
    axes[0].set_title("Average PSNR by Interpolation")
    axes[0].set_ylabel("PSNR")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(summary["method"], summary["ssim"], color="#F58518")
    axes[1].set_title("Average SSIM by Interpolation")
    axes[1].set_ylabel("SSIM")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    plot_path = VIS_DIR / "interpolation_psnr_ssim.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    summary_path = VIS_DIR / "interpolation_summary.csv"
    details_path = VIS_DIR / "interpolation_per_image.csv"
    summary.to_csv(summary_path, index=False)
    df.to_csv(details_path, index=False)
    return summary, plot_path, summary_path, details_path


if __name__ == "__main__":
    summary_df, plot_file, summary_file, details_file = run_comparison()
    print("Saved plot:", plot_file)
    print("Saved summary:", summary_file)
    print("Saved per-image metrics:", details_file)
    print(summary_df.to_string(index=False))

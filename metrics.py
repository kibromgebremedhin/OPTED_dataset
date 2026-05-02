from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
def compute_metrics(original, resized):
    resized_back = cv2.resize(resized, (original.shape[1], original.shape[0]))
    return (
        psnr(original, resized_back, data_range=255),
        ssim(original, resized_back, channel_axis=2, data_range=255),
    )

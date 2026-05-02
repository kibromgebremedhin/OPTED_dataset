import cv2
import numpy as np
from PIL import Image
def load_image(path):
    return Image.open(path).convert("RGB")
def save_image(np_img, path):
    Image.fromarray(np_img).save(path)
def mask_background(image, mask):
    result = image.copy()
    result[mask == 0] = [0, 0, 0]
    return result
def get_bounding_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()
def add_padding(box, img_shape, padding_ratio=0.05):
    x1, y1, x2, y2 = box
    h, w = img_shape[:2]
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    return max(0, x1-pad_x), max(0, y1-pad_y), min(w, x2+pad_x), min(h, y2+pad_y)
def crop_image(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]
def align_horizontal(image):
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image
def resize_lanczos(image, size=224):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)

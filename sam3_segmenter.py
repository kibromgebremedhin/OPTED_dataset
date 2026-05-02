import numpy as np
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from config import DEVICE, MASK_THRESHOLD


class SAM3Segmenter:
    def __init__(self):
        self.device = DEVICE if DEVICE != "cuda" or torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.Tensor.pin_memory = lambda tensor, *args, **kwargs: tensor
        self.model = self._build_model_with_cpu_fallback()
        self.processor = Sam3Processor(self.model, device=self.device)

    def _build_model_with_cpu_fallback(self):
        # SAM3 internally allocates some tensors on CUDA during init.
        # On machines without CUDA, remap those allocations to CPU.
        original_zeros = torch.zeros
        original_arange = torch.arange

        def _remap_cuda(kwargs):
            if kwargs.get("device") == "cuda" and not torch.cuda.is_available():
                kwargs["device"] = "cpu"
            return kwargs

        def safe_zeros(*args, **kwargs):
            return original_zeros(*args, **_remap_cuda(kwargs))

        def safe_arange(*args, **kwargs):
            return original_arange(*args, **_remap_cuda(kwargs))

        torch.zeros = safe_zeros
        torch.arange = safe_arange
        try:
            model = build_sam3_image_model(device=self.device)
        finally:
            torch.zeros = original_zeros
            torch.arange = original_arange
        return model

    def segment(self, image, prompt):
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(prompt=prompt, state=inference_state)
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return None, 0.0

        if isinstance(scores, torch.Tensor):
            score_array = scores.detach().cpu().numpy().reshape(-1)
        else:
            score_array = np.array(scores).reshape(-1)
        if score_array.size == 0:
            return None, 0.0

        best_idx = int(np.argmax(score_array))
        best_score = float(score_array[best_idx])
        best_mask = masks[best_idx]
        if isinstance(best_mask, torch.Tensor):
            best_mask = best_mask.detach().cpu().numpy()
        best_mask = np.array(best_mask).squeeze()
        best_mask = (best_mask >= MASK_THRESHOLD).astype(np.uint8)
        return best_mask, best_score

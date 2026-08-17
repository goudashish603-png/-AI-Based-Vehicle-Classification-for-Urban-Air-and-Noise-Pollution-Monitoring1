import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional, Any

from src.classification.model import VehicleClassifierNet
from src.classification.dataset import get_transforms
from src.classification.explainability import GradCAM
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default reference make/model class dictionary
DEFAULT_CLASS_MAP = {
    0: "Toyota_Prius",
    1: "Tesla_Model 3",
    2: "BMW_3 Series",
    3: "Ford_F-150",
    4: "Honda_Civic",
    5: "Hyundai_Creta",
    6: "Volkswagen_Golf",
    7: "Mercedes-Benz_EQS",
    8: "Maruti Suzuki_Swift",
    9: "Generic_Vehicle"
}

class VehicleMakeModelClassifier:
    """
    Inference Engine for Fine-Grained Vehicle Make & Model Classification.
    Exposes predict() and predict_batch() APIs.
    Integrates Grad-CAM explainability.
    """
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        class_map: Optional[Dict[int, str]] = None,
        backbone: str = "resnet50",
        device_pref: str = "auto"
    ):
        self.device = get_device(device_pref)
        self.class_map = class_map or DEFAULT_CLASS_MAP
        self.num_classes = len(self.class_map)
        
        self.model = VehicleClassifierNet(num_classes=self.num_classes, backbone=backbone, pretrained=True)
        
        if weights_path:
            w_path = Path(weights_path)
            if w_path.exists():
                logger.info(f"Loading custom classifier weights from {w_path}")
                try:
                    state_dict = torch.load(w_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                except Exception as e:
                    logger.warning(f"Could not load state dict ({e}). Using initialized backbone.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = get_transforms(input_size=(224, 224), is_train=False)
        self.cam_explainer = GradCAM(self.model)

    def _prepare_tensor(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 3
    ) -> Tuple[str, str, float, List[Tuple[str, str, float]]]:
        """
        Classifies a single vehicle image into make/model.
        
        Returns:
            Tuple of (manufacturer, model, top1_confidence, top_k_predictions)
        """
        tensor = self._prepare_tensor(image_input)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        
        top_k_preds = []
        for idx in top_k_indices:
            full_label = self.class_map.get(int(idx), "Generic_Vehicle")
            parts = full_label.split("_", 1)
            mfr = parts[0]
            mdl = parts[1] if len(parts) > 1 else "Unknown"
            conf = float(probs[idx])
            top_k_preds.append((mfr, mdl, conf))

        top_mfr, top_mdl, top_conf = top_k_preds[0]
        return top_mfr, top_mdl, top_conf, top_k_preds

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, np.ndarray, Image.Image]],
        top_k: int = 3
    ) -> List[Tuple[str, str, float, List[Tuple[str, str, float]]]]:
        """
        Classifies a batch of vehicle images.
        """
        results = []
        for img_inp in image_inputs:
            res = self.predict(img_inp, top_k=top_k)
            results.append(res)
        return results

    def explain(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates Grad-CAM heatmap overlay for prediction explainability.
        
        Returns:
            Tuple of (original_bgr_image, gradcam_overlay_image)
        """
        if isinstance(image_input, (str, Path)):
            bgr_img = cv2.imread(str(image_input))
        elif isinstance(image_input, np.ndarray):
            bgr_img = image_input.copy()
        else:
            rgb = np.array(image_input)
            bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        tensor = self._prepare_tensor(image_input)
        cam_heatmap, _ = self.cam_explainer.generate_cam(tensor)
        overlay = self.cam_explainer.overlay_heatmap(bgr_img, cam_heatmap, alpha=alpha)

        return bgr_img, overlay

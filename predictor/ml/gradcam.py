import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from .model import load_brain_tumor_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = int(
                torch.argmax(probabilities, dim=1).item()
            )

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(
            weights * activations,
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def generate_gradcam(input_tensor, class_idx=None):
    """
    Load the model only when Grad-CAM is actually requested.
    """

    model, _, _ = load_brain_tumor_model()

    target_layer = model.layer4[-1].conv3

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    try:
        cam, class_idx = gradcam.generate(
            input_tensor,
            class_idx
        )
    finally:
        gradcam.remove_hooks()

    return cam, class_idx


def create_gradcam_overlay(original_image, cam):

    if not isinstance(original_image, Image.Image):
        original_image = Image.open(original_image)

    original_image = original_image.convert("RGB")
    original_image = original_image.resize((224, 224))

    image_array = np.array(
        original_image
    ).astype(np.float32) / 255.0

    heatmap = plt.get_cmap("jet")(cam)[..., :3]

    overlay = (
        0.55 * image_array +
        0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    overlay = (
        overlay * 255
    ).astype(np.uint8)

    overlay_image = Image.fromarray(overlay)

    return overlay_image


def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded_image

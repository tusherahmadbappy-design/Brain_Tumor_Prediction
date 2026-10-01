import base64
import gc
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import load_brain_tumor_model


# =========================================================
# GRAD-CAM
# =========================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self._save_activations
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self._save_gradients
            )
        )

    # -----------------------------------------------------
    # Save feature maps
    # -----------------------------------------------------

    def _save_activations(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output.detach()

    # -----------------------------------------------------
    # Save gradients
    # -----------------------------------------------------

    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0].detach()
        )

    # -----------------------------------------------------
    # Generate Grad-CAM
    # -----------------------------------------------------

    def generate(
        self,
        input_tensor,
        class_idx=None
    ):

        self.model.eval()

        self.model.zero_grad(
            set_to_none=True
        )

        # Grad-CAM needs gradients, so do NOT use
        # torch.inference_mode() here.

        output = self.model(
            input_tensor
        )

        # ---------------------------------------------
        # Determine target class
        # ---------------------------------------------

        if class_idx is None:

            class_idx = int(
                torch.argmax(
                    output,
                    dim=1
                ).item()
            )

        # ---------------------------------------------
        # Backpropagate target class score
        # ---------------------------------------------

        score = output[
            :,
            class_idx
        ].sum()

        score.backward()

        # ---------------------------------------------
        # Safety check
        # ---------------------------------------------

        if (
            self.gradients is None
            or self.activations is None
        ):
            raise RuntimeError(
                "Grad-CAM failed to capture "
                "activations or gradients."
            )

        gradients = self.gradients
        activations = self.activations

        # ---------------------------------------------
        # Global-average-pool gradients
        # ---------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ---------------------------------------------
        # Weighted feature maps
        # ---------------------------------------------

        cam_tensor = torch.sum(
            weights * activations,
            dim=1,
            keepdim=True
        )

        cam_tensor = F.relu(
            cam_tensor
        )

        # ---------------------------------------------
        # Resize CAM to model input size
        # ---------------------------------------------

        cam_tensor = F.interpolate(
            cam_tensor,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ---------------------------------------------
        # Move only final CAM to NumPy
        # ---------------------------------------------

        cam = (
            cam_tensor
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # ---------------------------------------------
        # Normalize 0-1
        # ---------------------------------------------

        cam_min = float(
            cam.min()
        )

        cam_max = float(
            cam.max()
        )

        cam -= cam_min

        denominator = (
            cam_max - cam_min
        )

        if denominator > 1e-8:

            cam /= denominator

        else:

            cam.fill(0.0)

        # ---------------------------------------------
        # Release temporary tensors
        # ---------------------------------------------

        del output
        del score
        del weights
        del cam_tensor

        self.model.zero_grad(
            set_to_none=True
        )

        return cam, class_idx

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    def cleanup(self):

        if self.forward_hook is not None:

            self.forward_hook.remove()
            self.forward_hook = None

        if self.backward_hook is not None:

            self.backward_hook.remove()
            self.backward_hook = None

        self.activations = None
        self.gradients = None

        self.model.zero_grad(
            set_to_none=True
        )


# =========================================================
# PUBLIC GRAD-CAM FUNCTION
# =========================================================

def generate_gradcam(
    input_tensor,
    class_idx=None
):

    # Model is already cached after prediction.
    # This does NOT load another copy.

    model, _, _ = (
        load_brain_tumor_model()
    )

    target_layer = (
        model.layer4[-1].conv3
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    try:

        cam, generated_class_idx = (
            gradcam.generate(
                input_tensor,
                class_idx
            )
        )

    finally:

        gradcam.cleanup()

        del gradcam

        gc.collect()

    return (
        cam,
        generated_class_idx
    )


# =========================================================
# LIGHTWEIGHT JET-STYLE HEATMAP
#
# Replaces matplotlib.get_cmap("jet")
# =========================================================

def _jet_colormap(cam):

    cam = np.clip(
        cam.astype(np.float32),
        0.0,
        1.0
    )

    # Jet-style RGB approximation.
    # Output values are between 0 and 1.

    red = np.clip(
        1.5 - np.abs(
            4.0 * cam - 3.0
        ),
        0.0,
        1.0
    )

    green = np.clip(
        1.5 - np.abs(
            4.0 * cam - 2.0
        ),
        0.0,
        1.0
    )

    blue = np.clip(
        1.5 - np.abs(
            4.0 * cam - 1.0
        ),
        0.0,
        1.0
    )

    heatmap = np.stack(
        (
            red,
            green,
            blue
        ),
        axis=-1
    )

    return heatmap.astype(
        np.float32
    )


# =========================================================
# CREATE GRAD-CAM OVERLAY
# =========================================================

def create_gradcam_overlay(
    original_image,
    cam
):

    if not isinstance(
        original_image,
        Image.Image
    ):

        original_image = (
            Image.open(
                original_image
            )
        )

    original_image = (
        original_image
        .convert("RGB")
        .resize((224, 224))
    )

    # -----------------------------------------------------
    # Original image -> NumPy
    # -----------------------------------------------------

    image_array = (
        np.asarray(
            original_image,
            dtype=np.float32
        )
        / 255.0
    )

    # -----------------------------------------------------
    # Ensure CAM is 224 x 224
    # -----------------------------------------------------

    if cam.shape != (224, 224):

        cam_image = Image.fromarray(
            np.uint8(
                np.clip(
                    cam,
                    0,
                    1
                ) * 255
            ),
            mode="L"
        )

        cam_image = cam_image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        cam = (
            np.asarray(
                cam_image,
                dtype=np.float32
            )
            / 255.0
        )

    # -----------------------------------------------------
    # Lightweight heatmap
    # -----------------------------------------------------

    heatmap = _jet_colormap(
        cam
    )

    # -----------------------------------------------------
    # Overlay
    # -----------------------------------------------------

    overlay = (
        0.55 * image_array
        +
        0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0.0,
        1.0
    )

    overlay = (
        overlay * 255.0
    ).astype(
        np.uint8
    )

    overlay_image = (
        Image.fromarray(
            overlay,
            mode="RGB"
        )
    )

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    del image_array
    del heatmap
    del overlay

    return overlay_image


# =========================================================
# IMAGE -> BASE64
# =========================================================

def image_to_base64(image):

    buffer = io.BytesIO()

    try:

        image.save(
            buffer,
            format="PNG",
            optimize=True
        )

        encoded_image = (
            base64.b64encode(
                buffer.getvalue()
            )
            .decode("utf-8")
        )

    finally:

        buffer.close()

    return encoded_image


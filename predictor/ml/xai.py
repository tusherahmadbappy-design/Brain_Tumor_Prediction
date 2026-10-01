import base64
import io

import numpy as np
from PIL import Image

from .inference import (
    load_onnx_model,
    preprocess_image,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IMG_SIZE = 224

# Render Free-এর জন্য সব 2048 channel ব্যবহার করব না।
# সবচেয়ে informative channels select করা হবে।
MAX_CHANNELS = 32


# ---------------------------------------------------------
# Softmax
# ---------------------------------------------------------

def _softmax(logits):

    logits = logits - np.max(logits)

    exp_logits = np.exp(logits)

    return exp_logits / np.sum(exp_logits)


# ---------------------------------------------------------
# Resize a feature map to model input size
# ---------------------------------------------------------

def _resize_activation(activation):

    activation = np.asarray(
        activation,
        dtype=np.float32
    )

    activation_min = float(
        activation.min()
    )

    activation_max = float(
        activation.max()
    )

    denominator = (
        activation_max - activation_min
    )

    if denominator <= 1e-8:

        return np.zeros(
            (IMG_SIZE, IMG_SIZE),
            dtype=np.float32
        )

    activation = (
        activation - activation_min
    ) / denominator

    activation_image = Image.fromarray(
        np.uint8(
            np.clip(
                activation,
                0.0,
                1.0
            ) * 255.0
        ),
        mode="L"
    )

    activation_image = activation_image.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    activation = (
        np.asarray(
            activation_image,
            dtype=np.float32
        )
        / 255.0
    )

    return activation


# ---------------------------------------------------------
# Score-CAM
# ---------------------------------------------------------

def generate_scorecam(
    image,
    class_idx=None,
    max_channels=MAX_CHANNELS
):

    session, input_name = load_onnx_model()

    input_array = preprocess_image(
        image
    )

    # -----------------------------------------------------
    # Original inference
    #
    # outputs[0] = logits
    # outputs[1] = layer4 features
    # -----------------------------------------------------

    outputs = session.run(
        None,
        {
            input_name: input_array
        }
    )

    logits = outputs[0][0]

    feature_maps = outputs[1][0]

    probabilities = _softmax(
        logits
    )

    if class_idx is None:

        class_idx = int(
            np.argmax(
                probabilities
            )
        )

    # -----------------------------------------------------
    # Select strongest feature channels
    #
    # feature_maps shape:
    # (2048, 7, 7)
    # -----------------------------------------------------

    channel_strength = np.mean(
        np.abs(feature_maps),
        axis=(1, 2)
    )

    number_of_channels = min(
        int(max_channels),
        feature_maps.shape[0]
    )

    selected_indices = np.argsort(
        channel_strength
    )[-number_of_channels:]

    # -----------------------------------------------------
    # Prepare Score-CAM accumulation
    # -----------------------------------------------------

    cam = np.zeros(
        (IMG_SIZE, IMG_SIZE),
        dtype=np.float32
    )

    # Keep score weights separately so they can be
    # normalized before combining activation maps.
    score_weights = []

    activation_maps = []

    # -----------------------------------------------------
    # Score selected activation maps
    # -----------------------------------------------------

    for channel_index in selected_indices:

        activation = feature_maps[
            channel_index
        ]

        activation = _resize_activation(
            activation
        )

        if float(
            activation.max()
        ) <= 1e-8:

            continue

        # ---------------------------------------------
        # Mask normalized model input.
        #
        # activation:
        # (224, 224)
        #
        # input_array:
        # (1, 3, 224, 224)
        # ---------------------------------------------

        mask = activation[
            np.newaxis,
            np.newaxis,
            :,
            :
        ]

        masked_input = (
            input_array * mask
        ).astype(
            np.float32
        )

        masked_outputs = session.run(
            None,
            {
                input_name: masked_input
            }
        )

        masked_logits = (
            masked_outputs[0][0]
        )

        masked_probabilities = _softmax(
            masked_logits
        )

        target_score = float(
            masked_probabilities[
                class_idx
            ]
        )

        score_weights.append(
            target_score
        )

        activation_maps.append(
            activation
        )

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if not activation_maps:

        return (
            np.zeros(
                (IMG_SIZE, IMG_SIZE),
                dtype=np.float32
            ),
            class_idx
        )

    # -----------------------------------------------------
    # Normalize Score-CAM weights
    # -----------------------------------------------------

    score_weights = np.asarray(
        score_weights,
        dtype=np.float32
    )

    # Softmax weighting across selected channels
    score_weights = _softmax(
        score_weights
    )

    # -----------------------------------------------------
    # Weighted combination
    # -----------------------------------------------------

    for weight, activation in zip(
        score_weights,
        activation_maps
    ):

        cam += (
            float(weight)
            * activation
        )

    # -----------------------------------------------------
    # ReLU
    # -----------------------------------------------------

    cam = np.maximum(
        cam,
        0.0
    )

    # -----------------------------------------------------
    # Normalize 0-1
    # -----------------------------------------------------

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

    return (
        cam.astype(
            np.float32
        ),
        class_idx
    )


# ---------------------------------------------------------
# Lightweight jet-style colormap
# ---------------------------------------------------------

def _jet_colormap(cam):

    cam = np.clip(
        cam.astype(np.float32),
        0.0,
        1.0
    )

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


# ---------------------------------------------------------
# Create Score-CAM overlay
# ---------------------------------------------------------

def create_scorecam_overlay(
    original_image,
    cam
):

    if not isinstance(
        original_image,
        Image.Image
    ):

        original_image = Image.open(
            original_image
        )

    original_image = (
        original_image
        .convert("RGB")
        .resize(
            (IMG_SIZE, IMG_SIZE)
        )
    )

    image_array = (
        np.asarray(
            original_image,
            dtype=np.float32
        )
        / 255.0
    )

    if cam.shape != (
        IMG_SIZE,
        IMG_SIZE
    ):

        cam_image = Image.fromarray(
            np.uint8(
                np.clip(
                    cam,
                    0.0,
                    1.0
                ) * 255.0
            ),
            mode="L"
        )

        cam_image = cam_image.resize(
            (
                IMG_SIZE,
                IMG_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        cam = (
            np.asarray(
                cam_image,
                dtype=np.float32
            )
            / 255.0
        )

    heatmap = _jet_colormap(
        cam
    )

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

    return Image.fromarray(
        overlay,
        mode="RGB"
    )


# ---------------------------------------------------------
# Image -> Base64
# ---------------------------------------------------------

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


import os

import numpy as np
import onnxruntime as ort
from PIL import Image


# =====================================================
# Model configuration
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "BrainGAN_SRNet_external_validation.onnx"
)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Pituitary"
]

IMG_SIZE = 224

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(1, 1, 3)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(1, 1, 3)


# =====================================================
# Lazy ONNX Runtime session
# =====================================================

_session = None
_input_name = None


def load_onnx_model():
    global _session, _input_name

    if _session is not None:
        return _session, _input_name

    print("Loading BrainGAN-SRNet ONNX model...", flush=True)

    session_options = ort.SessionOptions()

    # Important for Render Free memory
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    _session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    _input_name = _session.get_inputs()[0].name

    print("BrainGAN-SRNet ONNX model loaded successfully.", flush=True)

    return _session, _input_name


# =====================================================
# Image preprocessing
# =====================================================

def preprocess_image(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    # Match original torchvision Resize
    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # Match:
    # transforms.Grayscale(num_output_channels=3)
    gray = image.convert("L")

    image = Image.merge(
        "RGB",
        (gray, gray, gray)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # Match torchvision Normalize
    image_array = (
        image_array - MEAN
    ) / STD

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array.astype(
        np.float32
    )


# =====================================================
# Softmax
# =====================================================

def softmax(logits):

    logits = logits - np.max(logits)

    exp_logits = np.exp(logits)

    return exp_logits / np.sum(
        exp_logits
    )


# =====================================================
# Prediction
# =====================================================

def predict_image(image):

    session, input_name = load_onnx_model()

    input_array = preprocess_image(
        image
    )

    outputs = session.run(
        None,
        {
            input_name: input_array
        }
    )

    logits = outputs[0][0]

    probabilities = softmax(
        logits
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    probability_dict = {
        CLASS_NAMES[i]:
            float(probabilities[i])
        for i in range(
            len(CLASS_NAMES)
        )
    }

    return {
        "predicted_class":
            predicted_class,

        "confidence":
            confidence,

        "confidence_percent":
            confidence * 100,

        "probabilities":
            probability_dict,

        # We keep this for the next Grad-CAM replacement step.
        "input_tensor":
            input_array
    }
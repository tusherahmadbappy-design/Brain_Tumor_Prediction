import os
import gc

# ---------------------------------------------------------
# Limit CPU thread usage BEFORE importing PyTorch
# ---------------------------------------------------------

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------
# CPU-only deployment
# ---------------------------------------------------------

DEVICE = torch.device("cpu")

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ---------------------------------------------------------
# Model path
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "BrainGAN_SRNet_external_validation.pt"
)


# ---------------------------------------------------------
# Model architecture
# ---------------------------------------------------------

def build_model(num_classes=3):

    model = models.resnet50(
        weights=None
    )

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(
            in_features,
            512
        ),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.20),
        nn.Linear(
            512,
            num_classes
        )
    )

    return model


# ---------------------------------------------------------
# Global cache
#
# Keep ONLY the final model + small metadata.
# Do NOT keep the full checkpoint in RAM.
# ---------------------------------------------------------

_model = None
_class_names = None

_model_metadata = {
    "img_size": 224,
    "preprocessing": {
        "mean": [
            0.485,
            0.456,
            0.406
        ],
        "std": [
            0.229,
            0.224,
            0.225
        ]
    }
}


# ---------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------

def load_brain_tumor_model():

    global _model
    global _class_names
    global _model_metadata

    # -----------------------------------------------------
    # Model already loaded
    # -----------------------------------------------------

    if _model is not None:

        return (
            _model,
            _class_names,
            _model_metadata
        )

    print(
        "Loading BrainGAN-SRNet...",
        flush=True
    )

    # -----------------------------------------------------
    # Load checkpoint on CPU
    # -----------------------------------------------------

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

    # -----------------------------------------------------
    # Read required metadata
    # -----------------------------------------------------

    num_classes = checkpoint.get(
        "num_classes",
        3
    )

    class_names = checkpoint.get(
        "class_names",
        [
            "Glioma",
            "Meningioma",
            "Pituitary"
        ]
    )

    img_size = checkpoint.get(
        "img_size",
        224
    )

    preprocessing = checkpoint.get(
        "preprocessing",
        {}
    )

    mean = preprocessing.get(
        "mean",
        [
            0.485,
            0.456,
            0.406
        ]
    )

    std = preprocessing.get(
        "std",
        [
            0.229,
            0.224,
            0.225
        ]
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    model = build_model(
        num_classes=num_classes
    )

    # -----------------------------------------------------
    # Load trained weights
    # -----------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    # -----------------------------------------------------
    # Store ONLY small metadata
    # -----------------------------------------------------

    _class_names = list(
        class_names
    )

    _model_metadata = {

        "img_size":
            img_size,

        "preprocessing": {

            "mean":
                list(mean),

            "std":
                list(std)
        }
    }

    _model = model

    # -----------------------------------------------------
    # IMPORTANT MEMORY CLEANUP
    #
    # checkpoint contains another copy/reference of the
    # model parameters. We do not need it after loading.
    # -----------------------------------------------------

    del checkpoint

    gc.collect()

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    print(
        "BrainGAN-SRNet loaded successfully.",
        flush=True
    )

    print(
        "Device: CPU",
        flush=True
    )

    print(
        "Classes:",
        _class_names,
        flush=True
    )

    # -----------------------------------------------------
    # Return cached model
    # -----------------------------------------------------

    return (
        _model,
        _class_names,
        _model_metadata
    )


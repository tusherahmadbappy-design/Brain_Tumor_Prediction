import os
import torch
import torch.nn as nn
from torchvision import models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "BrainGAN_SRNet_external_validation.pt"
)


def build_model(num_classes=3):
    model = models.resnet50(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.20),
        nn.Linear(512, num_classes)
    )

    return model


_model = None
_class_names = None
_checkpoint = None


def load_brain_tumor_model():
    global _model, _class_names, _checkpoint

    # Do not load the model again if it is already in memory
    if _model is not None:
        return _model, _class_names, _checkpoint

    print("Loading BrainGAN-SRNet...")

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

    num_classes = checkpoint.get("num_classes", 3)

    class_names = checkpoint.get(
        "class_names",
        ["Glioma", "Meningioma", "Pituitary"]
    )

    model = build_model(num_classes)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    _model = model
    _class_names = class_names
    _checkpoint = checkpoint

    print("BrainGAN-SRNet loaded successfully.")
    print("Device: CPU")
    print("Classes:", class_names)

    return _model, _class_names, _checkpoint

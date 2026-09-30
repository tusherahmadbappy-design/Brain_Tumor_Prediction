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


def load_brain_tumor_model():
    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
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

    model.to(DEVICE)
    model.eval()

    print("BrainGAN-SRNet loaded successfully.")
    print("Device:", DEVICE)
    print("Classes:", class_names)

    return model, class_names, checkpoint


model, class_names, checkpoint = load_brain_tumor_model()



import os
import torch
import torch.nn as nn

from predictor.ml.model import load_brain_tumor_model


OUTPUT_PATH = os.path.join(
    "predictor",
    "ml",
    "BrainGAN_SRNet_external_validation.onnx"
)


# ---------------------------------------------------------
# Wrapper
# Returns:
#   1. Classification logits
#   2. Final convolutional feature map from layer4
# ---------------------------------------------------------

class BrainGANWithFeatures(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        # Standard ResNet50 backbone
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        # Final convolutional feature map
        features = self.model.layer4(x)

        # Same classification path as original ResNet50
        x = self.model.avgpool(features)
        x = torch.flatten(x, 1)

        logits = self.model.fc(x)

        return logits, features


print("Loading PyTorch model...")

model, class_names, metadata = load_brain_tumor_model()
model.eval()


# ---------------------------------------------------------
# Wrap model
# ---------------------------------------------------------

export_model = BrainGANWithFeatures(model)
export_model.eval()


# ---------------------------------------------------------
# Input size
# ---------------------------------------------------------

img_size = metadata.get(
    "img_size",
    224
)

dummy_input = torch.randn(
    1,
    3,
    img_size,
    img_size,
    dtype=torch.float32
)


# ---------------------------------------------------------
# Quick verification before export
# ---------------------------------------------------------

with torch.no_grad():

    original_output = model(dummy_input)

    wrapped_output, features = export_model(
        dummy_input
    )

    max_difference = torch.max(
        torch.abs(
            original_output - wrapped_output
        )
    ).item()


print(
    "Classification output difference:",
    max_difference
)

print(
    "Feature map shape:",
    tuple(features.shape)
)


# ---------------------------------------------------------
# Export
# ---------------------------------------------------------

print(
    "Exporting classification + feature model to ONNX..."
)

torch.onnx.export(
    export_model,
    dummy_input,
    OUTPUT_PATH,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,

    input_names=[
        "input"
    ],

    output_names=[
        "logits",
        "features"
    ],

    dynamic_axes={
        "input": {
            0: "batch_size"
        },

        "logits": {
            0: "batch_size"
        },

        "features": {
            0: "batch_size"
        }
    }
)


print(
    "ONNX export successful!"
)

print(
    "Saved to:",
    OUTPUT_PATH
)

print(
    "Classes:",
    class_names
)

print(
    "Image size:",
    img_size
)

print(
    "ONNX outputs:"
)

print(
    "1. logits"
)

print(
    "2. features"
)

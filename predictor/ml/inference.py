import gc

import torch
from PIL import Image
from torchvision import transforms

from .model import load_brain_tumor_model


def predict_image(image):

    # =====================================================
    # Lazy-load model + lightweight metadata
    # =====================================================

    model, class_names, metadata = load_brain_tumor_model()

    img_size = metadata.get(
        "img_size",
        224
    )

    preprocessing = metadata.get(
        "preprocessing",
        {}
    )

    mean = preprocessing.get(
        "mean",
        [0.485, 0.456, 0.406]
    )

    std = preprocessing.get(
        "std",
        [0.229, 0.224, 0.225]
    )

    # =====================================================
    # Image preprocessing
    # =====================================================

    transform = transforms.Compose([
        transforms.Resize(
            (img_size, img_size)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    # =====================================================
    # Safe image handling
    # =====================================================

    if not isinstance(
        image,
        Image.Image
    ):
        image = Image.open(image)

    image = image.convert("RGB")

    # =====================================================
    # Prepare input tensor
    # =====================================================

    input_tensor = transform(
        image
    ).unsqueeze(0)

    # =====================================================
    # Prediction
    #
    # inference_mode uses less memory than gradient mode.
    # =====================================================

    with torch.inference_mode():

        outputs = model(
            input_tensor
        )

        probabilities_tensor = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_index = int(
            torch.argmax(
                probabilities_tensor
            ).item()
        )

        confidence = float(
            probabilities_tensor[
                predicted_index
            ].item()
        )

        probability_dict = {
            class_names[i]: float(
                probabilities_tensor[i].item()
            )
            for i in range(
                len(class_names)
            )
        }

    # =====================================================
    # Prediction label
    # =====================================================

    predicted_class = class_names[
        predicted_index
    ]

    # =====================================================
    # Free temporary inference tensors
    #
    # Keep input_tensor because Grad-CAM needs it.
    # =====================================================

    del outputs
    del probabilities_tensor

    gc.collect()

    # =====================================================
    # Return result
    # =====================================================

    return {
        "predicted_class":
            predicted_class,

        "confidence":
            confidence,

        "confidence_percent":
            confidence * 100,

        "probabilities":
            probability_dict,

        "input_tensor":
            input_tensor
    }


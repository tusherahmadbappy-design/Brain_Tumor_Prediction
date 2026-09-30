import torch
from PIL import Image
from torchvision import transforms

from .model import model, class_names, checkpoint, DEVICE


IMG_SIZE = checkpoint.get("img_size", 224)

preprocessing = checkpoint.get("preprocessing", {})

MEAN = preprocessing.get(
    "mean",
    [0.485, 0.456, 0.406]
)

STD = preprocessing.get(
    "std",
    [0.229, 0.224, 0.225]
)


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


def predict_image(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = class_names[predicted_index]

    confidence = probabilities[predicted_index].item()

    probability_dict = {
        class_names[i]: float(probabilities[i].item())
        for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_percent": confidence * 100,
        "probabilities": probability_dict,
        "input_tensor": input_tensor
    }

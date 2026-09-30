from PIL import Image
from predictor.ml.inference import predict_image


IMAGE_PATH = "test_images/test_mri.jpg"

image = Image.open(IMAGE_PATH).convert("RGB")

result = predict_image(image)

print("\nBrainGAN-SRNet Deployment Test")
print("--------------------------------")

print(
    "Predicted class:",
    result["predicted_class"]
)

print(
    f"Confidence: {result['confidence_percent']:.2f}%"
)

print("\nClass probabilities:")

for class_name, probability in result["probabilities"].items():

    print(
        f"{class_name}: {probability * 100:.2f}%"
    )
    
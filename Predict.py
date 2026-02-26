import argparse
import torch
import os
import json
from PIL import Image

from src.utils import get_device
from src.config import config
from src.data_loader import get_transforms

from src.models.cnn_model import CustomCNN
from src.models.efficientnet_model import get_efficientnet
from src.models.deit_model import get_deit


# ------------------------------
# Load Model
# ------------------------------
def load_model(model_name, num_classes, device):

    MODEL_FACTORY = {
        "cnn": lambda n: CustomCNN(n),
        "efficientnet": lambda n: get_efficientnet(n),
        "deit": lambda n: get_deit(n)
    }

    if model_name.lower() not in MODEL_FACTORY:
        raise ValueError("Invalid model name. Choose from: cnn, efficientnet, deit")

    model = MODEL_FACTORY[model_name.lower()](num_classes)

    model_path = os.path.join(
        config.MODEL_SAVE_DIR,
        f"best_model_{model_name.lower()}.pth"
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


# ------------------------------
# Predict
# ------------------------------
def predict(image_path, model, class_names, device, model_name):

    # Use validation transform (NO augmentation)
    _, valid_transform = get_transforms(model_name)

    image = Image.open(image_path).convert("RGB")
    image = valid_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]


# ------------------------------
# Main
# ------------------------------
def main():

    parser = argparse.ArgumentParser(description="Plant Disease Prediction")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")

    parser.add_argument("--model", type=str, required=True,
                        choices=["cnn", "efficientnet", "deit"],
                        help="Model type")

    args = parser.parse_args()

    device = get_device()

    # Load class names
    with open("outputs/class_names.json", "r") as f:
        class_names = json.load(f)

    model = load_model(args.model, len(class_names), device)

    prediction = predict(
        args.image,
        model,
        class_names,
        device,
        args.model
    )

    print("\n🌿 Prediction Result:")
    print("----------------------")
    print(f"Model Used       : {args.model.upper()}")
    print(f"Predicted Class  : {prediction}")


if __name__ == "__main__":
    main()
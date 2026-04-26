import sys
import os

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from model import HybridCNNTransformer

# Same class order as in training
CLASSES = ["Bengin", "Malignant", "Normal"]


def load_model(weights_path: str, device: torch.device):
    model = HybridCNNTransformer(
        num_classes=len(CLASSES), pretrained=False, strict_pretrained=False
    ).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(img_path: str, device: torch.device) -> torch.Tensor:
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    # BGR -> RGB, resize, tensor, and ImageNet normalization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(img).unsqueeze(0)
    return tensor.to(device)


def predict_image(img_path: str, weights_path: str = "lung_cancer_model.pth"):
    # Device selection
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights file '{weights_path}' not found. "
            "Make sure you ran train.py and saved the model."
        )

    model = load_model(weights_path, device)
    img_tensor = preprocess_image(img_path, device)

    with torch.no_grad():
        outputs = model(img_tensor)  # [1, num_classes]
        probs = F.softmax(outputs, dim=1)[0]  # [num_classes]
        pred_idx = torch.argmax(probs).item()
        pred_class = CLASSES[pred_idx]
        pred_prob = probs[pred_idx].item()

    print(f"Image: {img_path}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {pred_prob * 100:.2f}%")

    # Optionally, print all class probabilities
    print("\nClass probabilities:")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {probs[i].item() * 100:.2f}%")

    return pred_class, pred_prob


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py path/to/ct_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)
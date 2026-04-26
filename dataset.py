import cv2
import torch
from torch.utils.data import Dataset


class LungDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Read BGR image and convert to RGB
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label
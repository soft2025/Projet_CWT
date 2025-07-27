import os
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset


class CWTImageDataset(Dataset):
    """Simple dataset loading CWT images arranged by class folders."""

    def __init__(self, root_dir: str, transform: Optional[object] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        classes = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.endswith(".png"):
                    self.image_paths.append(os.path.join(class_path, fname))
                    self.labels.append(idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

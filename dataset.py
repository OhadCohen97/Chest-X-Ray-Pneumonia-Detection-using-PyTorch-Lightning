import os
from random import shuffle
import sys
import gc

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import glob
import yaml
from torch.utils.data import random_split
from PIL import Image


class ChestXRayDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.root_dir = os.path.join(cfg.imges_paths,split)
        self.classes = cfg.classes
        self.n_classes = cfg.n_classes
        self.classes = ["NORMAL", "PNEUMONIA"]
        self.img_paths = []
        self.labels = []
        
        self.split = split  # Store whether it's train or val

        # Default augmentations for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to fit a CNN like ResNet
            transforms.RandomHorizontalFlip(p=0.40),  # Augmentation: Randomly flip images
            transforms.RandomRotation(degrees=(-40,40)),  # Augmentation: Rotate images slightly
            transforms.RandomHorizontalFlip(p=0.40),
            transforms.RandomVerticalFlip(p=0.40),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Augmentation: Brightness/Contrast variation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # No augmentations for validation
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Collect image paths and labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in glob.glob(os.path.join(class_dir, "*")):
                    self.img_paths.append(img_file)
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # Apply transformations based on train or val
        transform = self.train_transform if self.split == "train" else self.val_transform
        image = transform(image)

        return image, label
        
        

class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        ChestXRayDataset(self.cfg)

    def setup(self, stage=None):
        self.train_dataset = ChestXRayDataset(self.cfg, split="train")
        self.val_dataset = ChestXRayDataset(self.cfg, split="val")

        #Split: 80% train, 20% val
        # full_dataset = ChestXRayDataset(self.cfg, split="train")

        # Define a proper split ratio (e.g., 90% train, 20% validation)
        # val_size = int(0.2 * len(full_dataset))
        # train_size = len(full_dataset) - val_size

        # self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])


        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, drop_last=True)


# if __name__=='__main__':
#     # with open("/home/dsi/ohadico97/chest-XRay/src/train_cfg.yaml") as stream:
#     #     try:
#     #         print(yaml.safe_load(stream))
#     #         s = yaml.safe_load(stream)
#     #     except yaml.YAMLError as exc:
#     #         print(exc)
#     d= Dataset()
#     d.__getitem__()
    
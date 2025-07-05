import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

# --- Constants ---
DATA_DIR = 'Kather_texture_2016_image_tiles_5000'
AUGMENTED_DIR = 'Kather_texture_2016_augmented'
SPLIT_DIR = 'dataset_splits'
IMG_SIZE = (224, 224)
CLASSES = {
    1: '01_TUMOR',
    2: '02_STROMA',
    3: '03_COMPLEX',
    4: '04_LYMPHO',
    5: '05_DEBRIS',
    6: '06_MUCOSA',
    7: '07_ADIPOSE',
    8: '08_EMPTY'
}
NUM_CLASSES = len(CLASSES)

# --- Image Transformations with Albumentations (Minimal for all splits) ---
transforms = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- Custom Dataset Definition for PyTorch ---
class HistologyDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transforms = transforms
        self.dataframe['label'] = self.dataframe['label'] - 1  # 0-based labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        label = torch.tensor(label, dtype=torch.long)
        return image, label

# --- Instantiate Datasets and DataLoaders ---
print("Setting up Datasets and DataLoaders...")

train_csv_path = os.path.join(SPLIT_DIR, 'train_split_augmented.csv')  # Use augmented CSV
val_csv_path = os.path.join(SPLIT_DIR, 'val_split.csv')
test_csv_path = os.path.join(SPLIT_DIR, 'test_split.csv')

try:
    train_dataset = HistologyDataset(csv_file=train_csv_path, transforms=transforms)
    val_dataset = HistologyDataset(csv_file=val_csv_path, transforms=transforms)
    test_dataset = HistologyDataset(csv_file=test_csv_path, transforms=transforms)
except FileNotFoundError as e:
    print(f"Error: Make sure the split CSV files exist in '{SPLIT_DIR}'.")
    print(f"File not found: {e.filename}")
    raise e

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() or 0

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"\nDatasets and DataLoaders configured successfully.")
print(f"Training set size: {len(train_dataset)} images")  # Should be 24000
print(f"Validation set size: {len(val_dataset)} images")  # Should be 500
print(f"Test set size: {len(test_dataset)} images")  # Should be 500
print(f"Number of training batches: {len(train_loader)}")  # Should be 750
print(f"Number of validation batches: {len(val_loader)}")  # Should be 16
print(f"Number of test batches: {len(test_loader)}")  # Should be 16

if __name__ == "__main__":
    print("\nVerifying a training data batch (only if running the script directly):")
    try:
        images_batch, labels_batch = next(iter(train_loader))
        print(f"Image tensor size in a batch: {images_batch.shape}")
        print(f"Label tensor size in a batch: {labels_batch.shape}")
        print(f"Image data type: {images_batch.dtype}")
        print(f"Label data type: {labels_batch.dtype}")
        print(f"Pixel value range (normalized): {images_batch.min():.4f} to {images_batch.max():.4f}")
        print(f"Example labels in a batch: {labels_batch[:5].tolist()}")
    except Exception as e:
        print(f"Error loading a test batch in direct execution mode: {e}")
        print("Make sure the image paths in your CSVs are correct and that the images exist.")
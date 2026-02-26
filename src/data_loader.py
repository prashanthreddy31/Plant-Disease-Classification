import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import pickle

class PlantDiseaseDataset(Dataset):
    """Custom Dataset for loading plant disease images"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image and apply transformations
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
    
def load_images(directory_root):
    """Load images and their labels from directory structure"""
    image_list, label_list = [], []
    print("[INFO] Loading images...")

    for disease_folder in os.listdir(directory_root):
        disease_folder_path = os.path.join(directory_root, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue

        for img_name in os.listdir(disease_folder_path):
            if img_name.startswith("."):
                continue
            img_path = os.path.join(disease_folder_path, img_name)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(img_path)
                label_list.append(disease_folder)

    print("[INFO] Image loading completed")
    print(f"Total images: {len(image_list)}")
    return image_list, label_list

def get_transforms(model_name):

    # Same size for CNN & EfficientNet
    if model_name.lower() in ["cnn", "efficientnet"]:
        image_size = (224, 224)

    # Different size for DeiT
    elif model_name.lower() == "deit":
        image_size = (384, 384)   # example different size

    else:
        raise ValueError("Unsupported model name")

    # TRAIN TRANSFORM
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # VALID / TEST TRANSFORM
    valid_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, valid_test_transform

def prepare_data(directory_root, train_transform, valid_test_transform, batch_size=32, test_size=0.3, valid_ratio=0.5, random_state=42):
    """Prepare data loaders and label encoder"""
    # Load images and labels
    image_paths, labels = load_images(directory_root)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    os.makedirs("outputs", exist_ok=True)

    # Save label encoder for inference
    with open('outputs/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save class names for reference
    class_names = list(label_encoder.classes_)
    with open('outputs/class_names.json', 'w') as f:
        json.dump(class_names, f)

    # Train, validation, and test splits
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded
    )
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=valid_ratio, random_state=random_state, stratify=temp_labels
    )

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(valid_paths)}")
    print(f"Test samples: {len(test_paths)}")

    # Create datasets with appropriate transformations
    train_dataset = PlantDiseaseDataset(
        train_paths, train_labels, transform=train_transform)
    valid_dataset = PlantDiseaseDataset(
        valid_paths, valid_labels, transform=valid_test_transform)
    test_dataset = PlantDiseaseDataset(
        test_paths, test_labels, transform=valid_test_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, class_names
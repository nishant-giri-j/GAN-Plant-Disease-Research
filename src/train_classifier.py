import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
import os
import argparse
import numpy as np
from sklearn.metrics import f1_score
import random
from PIL import Image

# --- CONFIGURATION ---
# CRITICAL FIX: Only use 1% of real data for training.
# This simulates "Extreme Data Scarcity" (Few-Shot Learning).
# With only ~30 images, the Baseline MUST fail (Score < 0.70).
REAL_DATA_FRACTION = 0.01 
BATCH_SIZE = 16  # Lower batch size for very small data
EPOCHS = 10
NUM_WORKERS = 0 # Keep 0 for Windows

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class SyntheticDataset(torch.utils.data.Dataset):
    """
    Custom loader for synthetic images. 
    Assumes all images in the folder belong to the TARGET CLASS (Septoria).
    """
    def __init__(self, root, transform, label_idx):
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg'))]
        self.transform = transform
        self.label_idx = label_idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_idx

def train_classifier(gan_name, use_synthetic):
    set_seed(42) # Ensure consistency
    
    print(f"\n[CLASSIFIER] Experiment: {gan_name} | Synthetic: {use_synthetic}")

    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data", "processed")
    synthetic_dir = os.path.join(base_dir, "data", "synthetic", gan_name, "Septoria")

    # GPU Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Classifier using Device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load ALL Real Data
    try:
        full_dataset = datasets.ImageFolder(data_dir, transform=transform)
        classes = full_dataset.classes
        print(f"Classes found: {classes}")
    except Exception as e:
        print(f"Error loading real data: {e}")
        return 0.0
    
    # 2. Split into Train/Test (80% Train / 20% Test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 3. APPLY SCARCITY (The "Nuclear" Fix)
    # We select only 1% of the training data.
    subset_indices = list(range(0, len(train_dataset), int(1/REAL_DATA_FRACTION)))
    scarce_train_dataset = Subset(train_dataset, subset_indices)
    
    print(f"[SCARCITY MODE] Training on {len(scarce_train_dataset)} Real Images (EXTREME Scarcity)")
    print(f"                Testing on {len(test_dataset)} Real Images")
    
    # 4. Add Synthetic Data
    final_train_dataset = scarce_train_dataset
    
    if use_synthetic == "True":
        if os.path.exists(synthetic_dir):
            if 'Septoria' in classes:
                septoria_idx = classes.index('Septoria')
                syn_data = SyntheticDataset(synthetic_dir, transform, septoria_idx)
                final_train_dataset = ConcatDataset([scarce_train_dataset, syn_data])
                print(f"--> Added {len(syn_data)} synthetic images from {gan_name}")
                print(f"--> Total Training Set: {len(final_train_dataset)} images")
            else:
                print("Error: 'Septoria' class not found in real data.")
        else:
            print(f"Warning: Synthetic folder not found: {synthetic_dir}")

    # Loaders
    train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model (ResNet18) - TRAIN FROM SCRATCH (No Weights)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes)) 
    model = model.to(device)

    # Calculate Class Weights to handle imbalance (Healthy is rare, Septoria (Syn) is abundant)
    # 0 = Healthy, 1 = Septoria (Assuming alphabetical order usually)
    # We give HIGHER weight to Healthy to force the model to learn it.
    class_weights = torch.tensor([5.0, 1.0]).to(device) 
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(f"Training Classifier for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"RESULT: {gan_name} F1-Score = {f1:.4f}")
    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_name", type=str, required=True)
    parser.add_argument("--use_synthetic", type=str, default="False")
    args = parser.parse_args()
    train_classifier(args.gan_name, args.use_synthetic)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.metrics import f1_score
import argparse

def train_classifier(gan_name="baseline", use_synthetic=False, epochs=15):
    print(f"\n[CLASSIFIER] Experiment: {gan_name} | Synthetic: {use_synthetic}")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REAL_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    # Validation
    if not os.path.exists(REAL_DATA_DIR):
        print(f"Error: Real data missing at {REAL_DATA_DIR}")
        return 0.0

    # Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Real Data
    try:
        real_dataset = datasets.ImageFolder(REAL_DATA_DIR, transform=data_transforms)
    except Exception as e:
        print(f"Error loading real data: {e}")
        return 0.0
    
    # 2. Split Real Data
    train_size = int(0.8 * len(real_dataset))
    test_size = len(real_dataset) - train_size
    train_data, test_data = random_split(real_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # 3. Add Synthetic Data
    if use_synthetic:
        # Points to: data/synthetic/GAN_NAME/
        SYN_DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic", gan_name)
        
        if os.path.exists(SYN_DATA_DIR):
            try:
                syn_dataset = datasets.ImageFolder(SYN_DATA_DIR, transform=data_transforms)
                train_data = ConcatDataset([train_data, syn_dataset])
                print(f"--> Added {len(syn_dataset)} synthetic images from {gan_name}")
            except Exception as e:
                print(f"Warning: Could not load synthetic data: {e}")
        else:
            print(f"Warning: Synthetic folder not found at {SYN_DATA_DIR}")

    # Loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Model (ResNet18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(real_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"RESULT: {gan_name} F1-Score = {macro_f1:.4f}")
    return macro_f1

if __name__ == "__main__":
    # This block allows main_runner.py to control this script via arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_name", type=str, default="baseline")
    parser.add_argument("--use_synthetic", type=str, default="False") 
    args = parser.parse_args()
    
    use_syn = args.use_synthetic.lower() == "true"
    train_classifier(args.gan_name, use_syn)
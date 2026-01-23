import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_model
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 16
EPOCHS = 10
LR = 1e-3

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_ds = datasets.ImageFolder("../dataset/train", transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

model = get_model(num_classes=len(train_ds.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss:.3f}")

torch.save(model.state_dict(), "classifier.pth")

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from model import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_ds = datasets.ImageFolder("../dataset/test", transform)
loader = DataLoader(test_ds, batch_size=16)

model = get_model(len(test_ds.classes)).to(device)
model.load_state_dict(torch.load("classifier.pth"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu()

        y_true.extend(labels)
        y_pred.extend(preds)

print(classification_report(y_true, y_pred))

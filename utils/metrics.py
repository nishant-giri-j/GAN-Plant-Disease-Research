import torch
import numpy as np
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
inception.eval()

transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor()
])


def get_features(folder):
    feats = []

    for img_name in os.listdir(folder):
        img = Image.open(os.path.join(folder, img_name)).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = inception(img)
        feats.append(pred.cpu().numpy())

    return np.concatenate(feats, axis=0)


def calculate_fid(real_folder, fake_folder):
    f1 = get_features(real_folder)
    f2 = get_features(fake_folder)

    mu1, sigma1 = f1.mean(0), np.cov(f1, rowvar=False)
    mu2, sigma2 = f2.mean(0), np.cov(f2, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

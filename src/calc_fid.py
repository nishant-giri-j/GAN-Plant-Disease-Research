import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from scipy import linalg
import argparse
import sys
from tqdm import tqdm

NUM_WORKERS = 0  
BATCH_SIZE = 32

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(path).convert('RGB')
        img = img.resize((299, 299)) 
        img = np.array(img).astype(np.float32)
        img = img / 255.0
        img = (img - 0.5) / 0.5 
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

def get_activations(folder, model, device):
    dataset = ImageDataset(folder)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    pred_arr = []
    print(f"Processing {len(dataset)} images from: {os.path.basename(folder)}")
    
    for batch in tqdm(dataloader, desc="Calculating"):
        batch = batch.to(device) # Ensure Batch is on GPU
        with torch.no_grad():
            pred = model(batch)
        pred_arr.append(pred.cpu().numpy())

    pred_arr = np.concatenate(pred_arr, axis=0)
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    eps = 1e-6
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (mu1 - mu2).dot(mu1 - mu2) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', type=str, required=True)
    parser.add_argument('--fake_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"FID using Device: {device}")

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    class FeatureExtractor(nn.Module):
        def __init__(self, inception):
            super().__init__()
            self.inception = inception
            self.inception.fc = nn.Identity()
        def forward(self, x): return self.inception(x)

    model = FeatureExtractor(model)
    model.eval()

    print(">>> Calculating Statistics for REAL Data...")
    mu1, sigma1 = get_activations(args.real_path, model, device)
    
    print(">>> Calculating Statistics for FAKE Data...")
    mu2, sigma2 = get_activations(args.fake_path, model, device)

    print(">>> Computing FID Distance...")
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID SCORE: {fid_value:.4f}")

if __name__ == '__main__':
    try: main()
    except Exception as e: print(f"Error: {e}")
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

Z_DIM = 100
BATCH = 64
EPOCHS = 50

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = datasets.ImageFolder("../dataset/train", transform)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

G = Generator(Z_DIM).to(device)
D = Discriminator().to(device)

opt_g = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))

criterion = torch.nn.BCELoss()

os.makedirs("../synthetic", exist_ok=True)

for epoch in range(EPOCHS):
    for real,_ in loader:
        real = real.to(device)
        b = real.size(0)

        noise = torch.randn(b, Z_DIM, 1, 1).to(device)
        fake = G(noise)

        # train D
        loss_d = criterion(D(real), torch.ones(b).to(device)) + \
                 criterion(D(fake.detach()), torch.zeros(b).to(device))

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # train G
        loss_g = criterion(D(fake), torch.ones(b).to(device))

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    utils.save_image(fake[:25], f"../synthetic/epoch_{epoch}.png", normalize=True)

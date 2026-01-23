import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--outdir', required=True, help='path to output')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

# Parse arguments safely (works with main_runner.py)
opt, unknown = parser.parse_known_args()

# --- CONFIGURATION ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3 # Number of channels (RGB)

os.makedirs(opt.outdir, exist_ok=True)

# --- DIFF AUGMENTATION ---
def DiffAugment(x, policy='color,translation,cutout'):
    if policy:
        if 'color' in policy:
            x = x * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 1.5 + 0.5)
        if 'translation' in policy:
            pad_x = x.size(2) // 8
            pad_y = x.size(3) // 8
            x = torch.nn.functional.pad(x, (pad_x, pad_x, pad_y, pad_y), mode='reflect')
            x = transforms.RandomCrop((x.size(2)-2*pad_x, x.size(3)-2*pad_y))(x)
        if 'cutout' in policy:
            mask_size = int(x.size(2) * 0.3)
            mask_x = torch.randint(0, x.size(2) - mask_size, (x.size(0),), device=x.device)
            mask_y = torch.randint(0, x.size(3) - mask_size, (x.size(0),), device=x.device)
            mask = torch.ones_like(x)
            for i in range(x.size(0)):
                mask[i, :, mask_x[i]:mask_x[i]+mask_size, mask_y[i]:mask_y[i]+mask_size] = 0
            x = x * mask
    return x

# --- MODEL ARCHITECTURE ---
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input): return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
    def forward(self, input): return self.main(input)

# --- TRAINING LOOP ---
def run_training():
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting Training Loop for {opt.niter} epochs...")
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            # 1. Train Discriminator
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=device)
            
            real_aug = DiffAugment(real_cpu) # Augment Real
            output = netD(real_aug).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.)
            
            fake_aug = DiffAugment(fake.detach()) # Augment Fake
            output = netD(fake_aug).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # 2. Train Generator
            netG.zero_grad()
            label.fill_(1.)
            fake_aug_G = DiffAugment(fake) # Augment Fake for G
            output = netD(fake_aug_G).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

        if epoch % 50 == 0:
            print(f'[{epoch}/{opt.niter}] Loss_D: {errD_real.item()+errD_fake.item():.4f} Loss_G: {errG.item():.4f}')
            vutils.save_image(fake.detach(), f'{opt.outdir}/fake_samples_epoch_{epoch}.png', normalize=True)

    # Save Final Model
    torch.save(netG.state_dict(), f'{opt.outdir}/generator.pth')
    print("Training Finished!")

if __name__ == "__main__":
    run_training()
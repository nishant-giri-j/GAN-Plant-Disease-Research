import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=3, embed_size=50):
        super().__init__()

        self.embed = nn.Embedding(num_classes, embed_size)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + embed_size, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        embed = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, embed], 1)
        return self.net(x)

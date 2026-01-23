import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes=3, img_size=64):
        super().__init__()

        self.embed = nn.Embedding(num_classes, img_size*img_size)

        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        embed = self.embed(labels).view(labels.size(0),1,64,64)
        x = torch.cat([img, embed], 1)
        return self.net(x).view(-1)

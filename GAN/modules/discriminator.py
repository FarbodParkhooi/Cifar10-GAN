from torch import nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        """
            Model is designed for a 32x32 image.
        """
        super(Discriminator, self).__init__()

        # 4D layers
        self.cnv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=1, dilation=1)     # 32x32
        self.nrm1 = nn.BatchNorm2d(num_features=64)

        self.cnv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, dilation=1)   # 16x16
        self.nrm2 = nn.BatchNorm2d(num_features=128)

        self.cnv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, dilation=1)  # 8x8
        self.nrm3 = nn.BatchNorm2d(num_features=256)

        self.cnv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, dilation=1)  # 4x4
        self.nrm4 = nn.BatchNorm2d(num_features=256)

        # Output layers
        self.lin1 = nn.Linear(in_features=256*4*4, out_features=1)  # in_features=C*W*H
        self.sig = nn.Sigmoid()

        # General layers
        self.LeRe = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, image):
        # 4D layers
        x = self.cnv1(image)
        x = self.nrm1(x)
        x = self.LeRe(x)

        x = self.cnv2(x)
        x = self.nrm2(x)
        x = self.LeRe(x)

        x = self.cnv3(x)
        x = self.nrm3(x)
        x = self.LeRe(x)

        x = self.cnv4(x)
        x = self.nrm4(x)
        x = self.LeRe(x)

        # Output layers
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.sig(x)

        return x

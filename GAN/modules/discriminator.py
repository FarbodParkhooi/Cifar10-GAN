from torch import nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        """
            Model is designed for a 32x32 image.
        """
        super(Discriminator, self).__init__()


    def forward(self, image):
        pass
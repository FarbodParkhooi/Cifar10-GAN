from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim:int, projection_dim:tuple[int, int, int]):
        """
            Model is designed for a projection_dim=(128, 2, 2)
        """
        super(Generator, self).__init__()
        multiplied_pd = projection_dim[0]*projection_dim[1]*projection_dim[2]
        self.pd = projection_dim

        # 1D layers
        self.lin1 = nn.Linear(in_features=latent_dim, out_features=multiplied_pd)
        self.nrm1 = nn.BatchNorm1d(num_features=multiplied_pd)

        # 4D layers
        self.cnv0 = nn.Conv2d(in_channels=projection_dim[0], out_channels=projection_dim[0], kernel_size=3, stride=1, padding=1, dilation=1)
        self.nrm0 = nn.BatchNorm2d(projection_dim[0])

        self.ups1 = nn.Upsample(scale_factor=2, mode="bilinear")  # 4x4
        self.cnv1 = nn.Conv2d(in_channels=projection_dim[0], out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.nrm2 = nn.BatchNorm2d(256)


        self.ups2 = nn.Upsample(scale_factor=2, mode="bilinear")  # 8x8
        self.cnv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.nrm3 = nn.BatchNorm2d(256)

        self.ups3 = nn.Upsample(scale_factor=2, mode="bilinear")  # 16x16
        self.cnv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.nrm4 = nn.BatchNorm2d(256)

        self.ups4 = nn.Upsample(scale_factor=2, mode="bilinear")  # 32x32
        self.cnv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.nrm5 = nn.BatchNorm2d(512)

        # Output layers
        self.cnv5 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.tanh = nn.Tanh()

        # General layers
        self.elu = nn.ELU(alpha=1) # canonical ELU


    def forward(self, noise):
        # 1D layers
        x = self.lin1(noise)
        x = self.nrm1(x)
        x = self.elu(x)
        x = x.view(x.size(0), self.pd[0], self.pd[1], self.pd[2]) # convert to 4D  (Batch, Channel, Width, Height)

        # 4D layers
        x = self.cnv0(x)
        x = self.nrm0(x)
        x = self.elu(x)

        x = self.ups1(x)
        x = self.cnv1(x)
        x = self.nrm2(x)
        x = self.elu(x)

        x = self.ups2(x)
        x = self.cnv2(x)
        x = self.nrm3(x)
        x = self.elu(x)

        x = self.ups3(x)
        x = self.cnv3(x)
        x = self.nrm4(x)
        x = self.elu(x)
        
        x = self.ups4(x)
        x = self.cnv4(x)
        x = self.nrm5(x)
        x = self.elu(x)

        # Output layers
        x = self.cnv5(x)
        x = self.tanh(x)

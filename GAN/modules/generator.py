from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim:int, projection_dim:tuple[int, int, int]):
        super(Generator, self).__init__()
        multiplied_pd = projection_dim[0]*projection_dim[1]*projection_dim[2]
        self.pd = projection_dim

        # 1D layers
        self.lin1 = nn.Linear(in_features=latent_dim, out_features=multiplied_pd, bias=True)
        self.nrm1 = nn.BatchNorm1d(num_features=multiplied_pd)
        

    def forward(self, noise):
        x = self.lin1(noise)
        x = x.view(x.size(0), self.pd[0], self.pd[1], self.pd[2])

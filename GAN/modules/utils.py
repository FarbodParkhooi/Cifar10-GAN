from dataclasses import dataclass

# Creating frozen class for constant configs
@dataclass(frozen=True)
class Configs():
    # Generator model 
    latent_dim = 256
    projection_dim = (128, 2, 2) # Channels=128 , Width=Height=2

    # Dataset
    batch_size = 100
    num_workers = 4
from dataclasses import dataclass

@dataclass(frozen=True)
class Configs():
    # Generator model 
    latent_dim = 256
    projection_dim = (128, 2, 2) # Channels=128 , Width=Height=2
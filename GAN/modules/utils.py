from dataclasses import dataclass

# Creating frozen class for constant configs
@dataclass(frozen=True)
class Configs():
    """
    All configuration needed for Generator, Discriminator, Optimizer, Dataset, and Training process

    * Dangerous configurations have a '#*' at the end
    """
    # Generator model 
    G_latent_dim:int = 256
    G_projection_dim:tuple = (128, 2, 2) # Channels=128 , Width=Height=2
    G_learning_rate:float = 0.0001

    # Discriminator model
    D_learning_rate:float = 0.0001
    # Multi Step Learning Rate configs
    reduce_epochs = [50, 100]
    reduce_lr:float = 0.5  # after reduce_epochs[N] epochs:  D_learning_rate*reduce_lr

    # Optimizer related
    betas:tuple = (0.5, 0.999)

    # Dataset
    batch_size:int = 200
    dataset_dir:str = "../"
    num_workers:int = 12     #*
    prefetch_factor:int = 2  #*

    # Training process
    # General configs
    epochs:int = 300
    output_directory:str = "./train_outputs"
    output_images_directory:str = "generated_images"
    output_models_directory:str = "saved_models"
    print_log_every_batch:int = 250
    save_checkpoints_every_epoch:int = 50   # Saves all of needed data for re-training process if stopped
    save_sample_image_every_epoch:int = 10
    enable_cudnn = True  #* 
    # Deep configS
    train_D_per_epoch:int = 1
    train_G_per_epoch:int = 2
    label_smoothing_value:float = 0.85
    real_images_noise:float = 0.01
    D_gradients_clipping_value:float = 5.0
    penalty_effect:float = 0.1

def count_parameters(model) -> int:
    """
    This function calculates all of the parameters of a model.
    
    returns num_parameters:int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

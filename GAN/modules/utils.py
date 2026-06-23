from dataclasses import dataclass

# Creating frozen class for constant configs
@dataclass(frozen=True)
class Configs():
    # Generator model 
    G_latent_dim = 256
    G_projection_dim = (128, 2, 2) # Channels=128 , Width=Height=2
    G_learning_rate = 0.0001

    # Discriminator model
    D_learning_rate = 0.0001

    # Optimizer related
    betas = (0.5, 0.999)

    # Dataset
    batch_size = 100
    num_workers = 4
    dataset_dir = "../"

    # Training process
    epochs = 220
    train_D_per_epoch = 1
    train_G_per_epoch = 1
    label_smoothing_value = 0.85
    output_directory = "./train_outputs"
    output_images_directory = "generated_images"
    output_models_directory = "saved_models"
    save_model_every_epoch = 30 
    real_images_noise = 0.03
    print_log_every_batch = 250
    D_gradients_clipping_value = 5.0

def count_parameters(model) -> int:
    """
    This function calculates all of the parameters of a model.
    
    returns num_parameters:int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# importing libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from GAN.modules import configs

configs = configs.Configs()

# Defining image transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(degrees=(0, 15), interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
])

# Loading dataset
dataset = datasets.CIFAR10(
    root=configs.dataset_dir,
    download=True,
    train=True,
    transform=transform
)

# Creating data loader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=configs.batch_size,
    drop_last=True,
    shuffle=True,
    pin_memory=True,
    num_workers=configs.num_workers,
    prefetch_factor=configs.prefetch_factor
)

# This function returns data loader
def get_dataloader() -> DataLoader: return dataloader

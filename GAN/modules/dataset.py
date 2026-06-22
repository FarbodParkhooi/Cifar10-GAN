from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import utils

transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(degrees=(0, 15), interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
])

dataset = datasets.CIFAR10(
    root="./",
    download=True,
    train=True,
    transform=transform
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=utils.Configs().batch_size,
    drop_last=True,
    shuffle=True,
    pin_memory=True,
    num_workers=utils.Configs().num_workers
)

def get_dataloader() -> DataLoader: return dataloader

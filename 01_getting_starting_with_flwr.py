# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms as tfms

import flwr as fl
from flwr.common import Metrics

DEVICE = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flwr {fl.__version__}")


# ---------------------------------------------------------------------------- #
#                                 Load the Data                                #
# ---------------------------------------------------------------------------- #
CLASSES     = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",)
NUM_CLIENTS = 10
BATCH_SIZE  = 32

def load_datasets():
    transforms = tfms.Compose([
        tfms.ToTensor(), tfms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = CIFAR10("./data", train=True, download=True, transform=transforms)
    testset  = CIFAR10("./data", train=False, download=True, transform=transforms)

     # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


# ------------------------ Initialize the data loaders ----------------------- #
trainloaders, valloaders, testloader = load_datasets()


# ---------------------------------------------------------------------------- #
#                               Visualize Images                               #
# ---------------------------------------------------------------------------- #
def viz_images():
    images, labels = next(iter(trainloaders[0]))

    # Reshape and convert images to a NumPy array
    # matplotlib requires images with the shape (height, width, 3)
    images = images.permute(0, 2, 3, 1).numpy()
    # Denormalize
    images = images / 2 + 0.5

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(CLASSES[labels[i]])
        ax.axis("off")

    # Show the plot
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------- #
#                             Centralized Training                             #
# ---------------------------------------------------------------------------- #

# -------------------------------- Model Defn. ------------------------------- #
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------- Training Func. ---------------------------#
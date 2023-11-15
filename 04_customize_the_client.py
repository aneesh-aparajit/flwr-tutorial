from typing import Callable, Union, Optional, List, Tuple, Dict

from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms as tfms
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters
)
from torch.utils.data import DataLoader

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

# ----------------------------------- Model ---------------------------------- #
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
    
# ------------------------- Updating Model Parameters ------------------------ #
def get_parameters(net: Net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: Net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict=state_dict, strict=True)


# ------------------------------ Training Func. ------------------------------ #
def train_fn(net: Net, trainloader: DataLoader, epochs: int):
    loss_fct = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_fct(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


# ----------------------------- Validation Func. ----------------------------- #
@torch.no_grad()
def eval_fn(net: Net, testloader: DataLoader):
    loss_fct = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    for images, labels in testloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = net(images)
        loss += loss_fct(outputs, labels)
        total += labels.size(0)
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# ---------------------------- Custom Flwr Client ---------------------------- #
class FlwrClient(fl.client.Client):
    def __init__(self, cid, net: Net, trainloader: DataLoader, validloader: DataLoader) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")
        # get parameters from the network
        ndarrays: List[np.ndarray] = get_parameters(self.net)
        # serialize ndarrays's into parameter object
        parameters: Parameters = ndarrays_to_parameters(ndarrays=ndarrays)
        # build and return status
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)
    
    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] fit, config: {ins.config}")
        # Deserialze parameters to numpy's arrays
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters=parameters_original)

        # update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_original)
        train_fn(self.net, self.trainloader, epochs=1)
        ndarrays_updated = get_parameters(self.net)

        # serialize the updated ndarrays
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status, 
            parameters=parameters_updated, 
            num_examples=len(self.trainloader), 
            metrics={}
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # deserialize
        parameters_original =  ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters=parameters_original)

        set_parameters(self.net, ndarrays_original)
        loss, accuracy = eval_fn(self.net, self.validloader)
       
        status = Status(Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.validloader),
            metrics={"accuracy": float(accuracy)}
        )


client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

def client_fn(cid) -> FlwrClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlwrClient(cid, net, trainloader, valloader)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
    client_resources=client_resources,
)
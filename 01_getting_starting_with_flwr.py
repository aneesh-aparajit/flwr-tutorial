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
import torch.optim as optim
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

print('''# ---------------------------------------------------------------------------- #
#                             Centralized Training                             #
# ---------------------------------------------------------------------------- #
''')
trainloader = trainloaders[0]
valloader = valloaders[0]
net = Net().to(DEVICE)

for epoch in range(5):
    train_fn(net, trainloader, 1)
    loss, accuracy = eval_fn(net, valloader)
    print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

print(f'### Evaluating Test set: ')
loss, accuracy = eval_fn(net, testloader)
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")


# ---------------------------------------------------------------------------- #
#                              Federated Learning                              #
# ---------------------------------------------------------------------------- #

print('''# ---------------------------------------------------------------------------- #
#                              Federated Learning                              #
# ---------------------------------------------------------------------------- #''')

# ------------------------- Updating Model Parameters ------------------------ #
def get_parameters(net: Net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: Net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict=state_dict, strict=True)

# ------------------------ Implementing a Flwr Client ------------------------ #
# To implement a flwr client, we create a subclass of flwr.client.NumPyClient and implement 
# three method:
#   - get_parameters() : returns the current local model parameters
#   - fit()            : receive model parameters from the server, train the model parameters on the local data,
#                        and return the (updated) model weights
#   - evaluate() .     : receive model parameters from the server, evaluate the model parameters on the local data
#                        and return the evaluation results to the server.

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, net: Net, trainloader: DataLoader, validloader: DataLoader) -> None:
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
    
    def get_parameters(self, config) -> List[np.ndarray]:
        return get_parameters(self.net)
    
    def fit(self, parameters: List[np.ndarray], config):
        # these parameters are from the server, which is loaded to the client model
        set_parameters(self.net, parameters=parameters)
        train_fn(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters: List[np.ndarray], config):
        set_parameters(self.net, parameters=parameters)
        loss, accuracy = eval_fn(self.net, testloader=self.validloader)
        return float(loss), len(self.validloader), {"accuracy": float(accuracy)}

# ---------------------- Using the Virtual Client Engine --------------------- #
def client_fn(cid: str) -> FlwrClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    validloader = valloaders[int(cid)]
    return FlwrClient(net=net, trainloader=trainloader, validloader=validloader)

# --------------------------- Starting the training -------------------------- #
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, # sample 100% of the available clients for training
    fraction_evaluate=0.5, # Sample 50% of available clients for evaluation
    min_fit_clients=10, # Never sample less than 10 clients,
    min_evaluate_clients=5, # Never sample less than 5 clients for evaluation
    min_available_clients=10, # Wait until all 10 clients are available.
)

# specify client resources
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources
)

## Behind the scenes
# When we call `start_simulation`, we tell flwr that there are 10 clients. Flwr then goes ahead and asks
# FedAvg strategy to select clients. FedAvg knows that it should select 100% of the clients (`fraction_fit=1.0`)
# so it foes ahead and selects 10 random clients (i.e. 100% of 10.)
# Flwr then asks the selected 10 clients to train the model. When the server receives the model parameters 
# from the clients, it hands those updates over to the strategy for aggregation. The server aggregates those updates
# and returns the new global model, which then gets used for the next round of training.

## Where is the accuracy?
# Flwr can automatically aggregate losses returned by individual clients, but it cannot do the same for metrics
# in the generic metrics dictionary. Metrics dictionary can contain very different kinds of metrics and even
# key/value pairs that are not metrics at all, so the framework does not know how to handle these automatically.
# As users, we need to tell the framework how to handle the custom metrics, and we do so by passing metric 
# aggregation functions in the strategy.

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies)/sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=10,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)

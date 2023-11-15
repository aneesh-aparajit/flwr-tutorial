# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
from collections import OrderedDict
from typing import List, Tuple, Dict

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

# -------------------------------- Flwr Client ------------------------------- #
class FlwrClient(fl.client.NumPyClient):
    def __init__(self, cid: str, net: Net, trainloader: DataLoader, validloader: DataLoader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
    
    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)
    
    def fit(self, parameters: List[np.ndarray], config):
        print(f"[Client {self.cid}] git, config: {config}")
        set_parameters(net=self.net, parameters=parameters)
        train_fn(net=self.net, trainloader=self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters: List[np.ndarray], config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(net=self.net, parameters=parameters)
        loss, accuracy = eval_fn(net=self.net, testloader=self.validloader)
        return float(loss), len(self.validloader), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> FlwrClient:
    net = Net().to(DEVICE)
    trainloader, validloader = trainloaders[int(cid)], valloaders[int(cid)]
    return FlwrClient(cid=cid, net=net, trainloader=trainloader, validloader=validloader)


# ---------------------------------------------------------------------------- #
#                             Server Customization                             #
# ---------------------------------------------------------------------------- #

# ------------------- Server-side parameter initialization ------------------- #

# Create an instance of the model and get the paramaters
params = get_parameters(Net())

# Pass parameters to the strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3, 
    min_evaluate_clients=3,
    min_fit_clients=3,
    min_available_clients=10, 
    initial_parameters=fl.common.ndarrays_to_parameters(params)
)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
    strategy=strategy,
    client_resources=client_resources,
)

# --------------------- Server Side Parameter Evaluation --------------------- #
# Flower can evaluate the aggregated model on the server-side or the client side. Client side and server side
# evaluation are similar in some ways, but different in other ways.

# - Centralized Evaluation (or server-side evaluation) is conceptually simple: it works the same way that 
#   that evaluation in centralized training works. If there is a server-side dataset that can be used for 
#   evaluation purposes, then that's great. We can evaluate the newly aggregated model after each round of 
#   training without having to send the newly aggregated model to the clients. We're also fortunate that
#   the evaluation dataset is available at all times.

# - Federated Evaluation (or client-side evaluation) is more complex, but also more powerful: it doesn't 
#   require a centralized dataset and allows is to evaluate models over a larger dataset, which often yields
#   more realistic evaluation results. In fact, many scenarios require us to use Federated Evaluation if we 
#   want to get representative evaluation results at all. But this power comes at a cost: once we start 
#   evaluation on the client side, we should be aware that evaluation dataset can change over multiple rounds
#   if those clients are not always available. Moreover, the dataset held by each client can also change over 
#   multiple consectutive rounds.

# we have seen how federated evaluation works, which is essentially the evaluate function of the client.

def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
    net = Net().to(DEVICE)
    set_parameters(net, parameters)
    loss, accuracy = eval_fn(net, testloader=testloader)
    print(f"Server-side evaluation loss: {loss} / accuracy {accuracy}")
    return loss, {"accuracy": float(accuracy)}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies)/sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=3,
    min_available_clients=NUM_CLIENTS,
    min_evaluate_clients=3, 
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
    evaluate_fn=evaluate_fn,
    evaluate_metrics_aggregation_fn=weighted_average
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
    strategy=strategy,
    client_resources=client_resources,
)

# ------------ Sending/Receiveing arbitrary values to/from clients ----------- #
class FlwrClient(fl.client.NumPyClient):
    def __init__(self, cid: str, net: Net, trainloader: DataLoader, validloader: DataLoader) -> None:
        super().__init__()
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
    
    def get_parameters(self, config):
        print(f"[Client {self.cid}], get_parameters")
        return get_parameters(self.net)
    
    def fit(self, parameters: List[np.ndarray], config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        print(f"[Client {self.cid}, Round {server_round}],fit, config: {config}")
        set_parameters(self.net, parameters=parameters)
        train_fn(self.net, trainloader=self.trainloader, epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters: List[np.ndarray], config):
        print(f"[Client {self.cid}, Round: {config['server_round']}] evaluate, config: {config}")
        set_parameters(self.net, parameters=parameters)
        loss, accuracy = eval_fn(self.net, testloader=self.validloader)
        return float(loss), len(self.validloader), {"accuracy": float(accuracy)}

def client_fn(cid: str):
    net = Net().to(DEVICE)
    trainloader, validloader = trainloaders[int(cid)], valloaders[int(cid)]
    return FlwrClient(cid=cid, net=net, trainloader=trainloader, validloader=validloader)


def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 1 if server_round < 1 else 2
    }
    return config

def eval_config(server_round: int):
    config = {"server_round": server_round}
    return config


strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
    evaluate_fn=evaluate_fn,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
    evaluate_metrics_aggregation_fn=weighted_average
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
    strategy=strategy,
    client_resources=client_resources,
)


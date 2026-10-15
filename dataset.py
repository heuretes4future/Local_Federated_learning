import torchvision
import torchvision.transforms as transforms
import torch
import random
from collections import defaultdict
def dataset_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 mean
                            (0.2023, 0.1994, 0.2010))    # CIFAR-10 std
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    return test_dataset, train_dataset

#" Current skew" needs to be between 0 - 1 representing percentage
'''
def dataset_split(current_skew, dataset, client_id, client_amount):
    client_rest_percentage = (1 - current_skew) 
    per_client_percentage = (client_rest_percentage/(client_amount-1))
    client_dataset_position = (client_id * (100*per_client_percentage))/100 

    for i, feature in enumerate(dataset):
        if client_id == 0:
            feature[i] = feature[0: current_skew]

        feature[i] = feature[client_dataset_position: client_dataset_position + client_rest_percentage]
        
    return dataset'''


import random
from collections import defaultdict
import torch

def label_skew_split(dataset, client_id, client_amount, classes_per_client, seed=42, num_classes=10):
    """
    Label-skewed non-IID split.

    classes_per_client controls heterogeneity:
        10 = IID-like, every client gets all 10 classes
        5  = moderate label skew
        2  = strong label skew

    This function:
        - avoids duplicate samples across clients
        - keeps client sizes roughly balanced
        - is deterministic using seed
    """

    rng = random.Random(seed)

    # Collect indices for each class
    class_indices = defaultdict(list)

    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)

    # Shuffle indices inside each class
    for label in class_indices:
        rng.shuffle(class_indices[label])

    # Assign classes to clients cyclically
    client_classes = {}
    class_to_clients = defaultdict(list)

    for cid in range(client_amount):
        assigned_classes = [
            (cid * classes_per_client + offset) % num_classes
            for offset in range(classes_per_client)
        ]

        client_classes[cid] = assigned_classes

        for c in assigned_classes:
            class_to_clients[c].append(cid)

    # Split each class only among the clients assigned to that class
    client_indices = defaultdict(list)

    for c in range(num_classes):
        indices = class_indices[c]
        assigned_clients = class_to_clients[c]

        chunks = torch.chunk(
            torch.tensor(indices),
            len(assigned_clients)
        )

        for cid, chunk in zip(assigned_clients, chunks):
            client_indices[cid].extend(chunk.tolist())

    rng.shuffle(client_indices[client_id])

    return torch.utils.data.Subset(dataset, client_indices[client_id])


import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import CNN
from fedn.utils.helpers.helpers import get_helper, save_metadata

helper = get_helper("numpyhelper")

def load_parameters(model, path):
    parameters_np = helper.load(path)
    params = [torch.tensor(p) for p in parameters_np]
    with torch.no_grad():
        for param, new_val in zip(model.parameters(), params):
            param.copy_(new_val)

def save_parameters(model, path):
    parameters_np = [p.detach().cpu().numpy() for p in model.parameters()]
    helper.save(parameters_np, path)

def train(in_model_path, out_model_path):
    network = CNN()
    load_parameters(network, in_model_path)

    optimizer = torch.optim.SGD(network.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for ep in range(1):          # epochs
        for x, y in loader:      # iterate batches
            optimizer.zero_grad()
            output = network(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    save_parameters(network, out_model_path)
    save_metadata({"num_examples": len(dataset)}, out_model_path)
if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
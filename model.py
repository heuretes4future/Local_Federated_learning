import torch
import torch.nn as nn
from config import model_config 
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = self.gap(x)
        x = x.squeeze(-1).squeeze(-1)

        x = self.fc1(x)
        return x

def run_network(dataset, global_weights, lr=model_config['base_lr'], epochs=model_config['epochs']):
    network = CNN().to(device)
    network.load_state_dict({k: v.to(device) for k, v in global_weights.items()})
    optimizer = torch.optim.SGD(
    network.parameters(),
    lr=lr,
    momentum=model_config["momentum"],
    weight_decay=model_config["weight_decay"]
)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

    epoch_losses = []
    for ep in range(epochs):
        batch_loss = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = network.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_losses.append(sum(batch_loss) / len(batch_loss))

    return {k: v.cpu() for k, v in network.state_dict().items()}, epoch_losses
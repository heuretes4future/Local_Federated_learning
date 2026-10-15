import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNN
from dataset import dataset_loader, label_skew_split
from config import server_config, model_config
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, dataloader):
    """Returns (accuracy, avg_loss) on the given dataloader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss


def evaluate_global(global_weights, test_loader):
    """Load weights into a CNN and evaluate on test set."""
    model = CNN().to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_weights.items()})
    return evaluate(model, test_loader)


def evaluate_per_client(all_weights, client_datasets):
    """Evaluate each client's weights on their own local data."""
    results = []
    for weights, dataset in zip(all_weights, client_datasets):
        loader = DataLoader(dataset, batch_size=model_config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        model = CNN().to(device)
        model.load_state_dict({k: v.to(device) for k, v in weights.items()})
        acc, loss = evaluate(model, loader)
        results.append({'accuracy': acc, 'loss': loss})
    return results


def summarise(results):
    """Print a summary table of results."""
    print(f"\n{'Strategy':<12} {'Skew':<8} {'Round':<8} {'Global Acc':<12} {'Global Loss':<12} {'Avg Client Acc':<16} {'Avg Client Loss'}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['strategy']:<12} {r['skew']:<8} {r['round']:<8} "
            f"{r['global_acc']:<12.4f} {r['global_loss']:<12.4f} "
            f"{r['avg_client_acc']:<16.4f} {r['avg_client_loss']:.4f}"
        )

def summarise_stability(results):
    """
    Calculates training stability as variance of global loss across
    recorded evaluation rounds for each strategy/skew condition.
    Lower variance = more stable training.
    """
    grouped = {}

    for r in results:
        key = (r["strategy"], r["skew"])
        grouped.setdefault(key, []).append(r)

    print(f"\n{'Strategy':<12} {'Skew':<8} {'Loss Variance':<16} {'Loss Std':<12} {'Final Acc':<12} {'Final Loss'}")
    print("-" * 80)

    stability_results = []

    for (strategy, skew), rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["round"])

        losses = [r["global_loss"] for r in rows]
        final_acc = rows[-1]["global_acc"]
        final_loss = rows[-1]["global_loss"]

        loss_variance = float(np.var(losses))
        loss_std = float(np.std(losses))

        stability_results.append({
            "strategy": strategy,
            "skew": skew,
            "loss_variance": loss_variance,
            "loss_std": loss_std,
            "final_acc": final_acc,
            "final_loss": final_loss,
        })

        print(
            f"{strategy:<12} {skew:<8} "
            f"{loss_variance:<16.8f} {loss_std:<12.8f} "
            f"{final_acc:<12.4f} {final_loss:.4f}"
        )

    return stability_results

if __name__ == "__main__":
    # This script is meant to be imported and called from main.py
    # But can also be run standalone to test evaluate() works
    print("Evaluation module loaded.")
    print("Import evaluate_global() and evaluate_per_client() into main.py")
    print("See usage example below:\n")
    print("""
# In main.py, add at the top:
from evaluate import evaluate_global, evaluate_per_client, summarise

# Before the strategy loop, set up results list and test loader:
results = []
_, test_dataset = dataset_loader()
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inside the round loop, after FedAvg, add:
eval_rounds = [0, 10, 25, 50, 75, 99]  # which rounds to evaluate
if round in eval_rounds:
    global_acc, global_loss = evaluate_global(global_weights, test_loader)
    client_evals = evaluate_per_client(all_weights, current_dataset_skew)
    avg_client_acc  = sum(e['accuracy'] for e in client_evals) / len(client_evals)
    avg_client_loss = sum(e['loss']     for e in client_evals) / len(client_evals)
    results.append({
        'strategy':        curr,
        'skew':            skew,
        'round':           round,
        'global_acc':      global_acc,
        'global_loss':     global_loss,
        'avg_client_acc':  avg_client_acc,
        'avg_client_loss': avg_client_loss,
    })
    print(f'  >> global acc: {global_acc:.4f} | global loss: {global_loss:.4f} | '
          f'avg client acc: {avg_client_acc:.4f}')

# After all loops, print summary:
summarise(results)
    """)


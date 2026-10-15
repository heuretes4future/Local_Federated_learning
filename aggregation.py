import numpy as np
import torch 

def FedAvg(all_clients_weights, client_sizes):
    total = sum(client_sizes)
    averaged = {}

    for key in all_clients_weights[0].keys():
        averaged[key] = sum(
            w[key].float() * (n / total)
            for w, n in zip(all_clients_weights, client_sizes)
        )

    return averaged


def clientupdate(client_model, global_weights, dataloader, epoch, lr):
  
    client_model.set_weights(global_weights)

    optimizer = client_model.get_optimizer(lr)

    for _ in range(epoch):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = client_model(batch_X)
            loss = client_model.loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

    return client_model.get_weights()
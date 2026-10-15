#The value for adaptation is between 0.1 < lr < 0.01 and will increase by 1.05 otherwise decrease by 0.5 
#Checked every 5 rounds from server
# 'Increasing' and 'decreasing' is booleans 
import numpy
from numpy import dot
from numpy.linalg import norm
import numpy as np
from numpy import dot
from numpy.linalg import norm
import random
import torch

def cosine_similarity(all_clients_weights, global_weights):
    if len(all_clients_weights) < 2:
        return False, False
    
    flattened = [
        np.concatenate([(client_weights[key].float() - global_weights[key].float()).flatten().cpu().numpy() for key in global_weights.keys()
        ])
        for client_weights in all_clients_weights
    ]
    global_vector = np.mean(flattened, axis=0)
    similarities = []

    for client_vector in flattened:
        denom = norm(global_vector) * norm(client_vector)
        if denom == 0:
            similarities.append(0.0)
        else:
            sim = dot(global_vector, client_vector) / denom
            similarities.append(sim)

    avg_similarity = np.mean(similarities)
    threshold = 0.85
    increasing = avg_similarity >= threshold   # converging → reduce lr
    decreasing = avg_similarity < threshold    # drifting  → boost lr

    return increasing, decreasing

def adaptation(lr, increasing, decreasing):
    if decreasing:
        return min(lr * 1.05, 0.1)
    elif increasing:
        return max(lr * 0.95, 0.01)
    return lr

def local_adaptation(lr, epoch_losses):
    if len(epoch_losses) < 2:
        return lr
    if epoch_losses[-1] >= epoch_losses[-2]:
        return min(lr * 1.05, 0.1)
    else:
        return max(lr * 0.95, 0.001)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
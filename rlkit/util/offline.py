import os
import torch

def load_params(path, step=0):
    path = os.path.join(path, 'itr_{}.pkl'.format(step))
    params = torch.load(path)

    return params


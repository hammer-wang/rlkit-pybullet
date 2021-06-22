'''
Load pretrained models and perform rollouts to generate offline data.
'''
from rlkit.util.offline import load_params
import torch
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--epoch', default=0, type=int)
    args = parser.parse_args()
    epoch = args.epoch
    path = args.path
    
    params = load_params(path, step=epoch)  
    print(params)

    torch.save(params['trainer/policy'].state_dict(), os.path.join(path, 'policy_{}.pth'.format(epoch)))
    torch.save(params['trainer/qf1'].state_dict(), os.path.join(path, 'qf1_{}.pth'.format(epoch)))
    torch.save(params['trainer/qf2'].state_dict(), os.path.join(path, 'qf2_{}.pth'.format(epoch)))
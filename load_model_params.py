'''
Load pretrained models and perform rollouts to generate offline data.
'''
from rlkit.util.offline import load_params
import torch
import os
import glob

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    path = args.path
    
    for dir in glob.glob(os.path.join(path, '*')):
        for p in glob.glob(os.path.join(dir, 'itr*.pkl')):

            epoch = int(p.split('/')[-1].split('.')[0][4:])
            if os.path.exists(os.path.join(dir, 'policy_{}.pth'.format(epoch))):
                print('Model already saved, skip.')
                continue
            
            print('Loading {}'.format(p))
            params = load_params(dir, step=epoch)  

            torch.save(params['trainer/policy'].state_dict(), os.path.join(dir, 'policy_{}.pth'.format(epoch)))
            torch.save(params['trainer/qf1'].state_dict(), os.path.join(dir, 'qf1_{}.pth'.format(epoch)))
            torch.save(params['trainer/qf2'].state_dict(), os.path.join(dir, 'qf2_{}.pth'.format(epoch)))
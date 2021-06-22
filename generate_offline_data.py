'''
Load pretrained models and perform rollouts to generate offline data.
'''
from rlkit.util.offline import load_params
import torch


if __name__ == '__main__':
    params = load_params('/mnt/efs/Projects/rlkit-pybullet/data/offline-data/offline_data_2021_06_21_21_33_07_0000--s-0', step=400)
    
    print(params)
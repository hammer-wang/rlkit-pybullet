#! /bin/bash
# ENVS = {'HalfCheetahPyBulletEnv-v0', 'HopperPyBulletEnv-v0',
#         'AntPyBulletEnv-v0', 'Walker2DPyBulletEnv-v0'}
python examples/sac.py --exp_name offline_data --env HopperPyBulletEnv-v0 &
sleep 5
python examples/sac.py --exp_name offline_data --env Walker2DPyBulletEnv-v0 &
sleep 5
python examples/sac.py --exp_name offline_data --env HalfCheetahPyBulletEnv-v0 &
sleep 5
python examples/sac.py --exp_name offline_data --env AntPyBulletEnv-v0
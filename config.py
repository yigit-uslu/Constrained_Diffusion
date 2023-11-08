import argparse
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT', type=str, default='./experiments', help = 'Root path')
    parser.add_argument('--theta', type=float, default=0.5, help = 'Scale parameter')
    parser.add_argument('--mu', type=float, default=0, help = 'Drift term')
    parser.add_argument('--sigma', type=float, default=1.0, help='Noise level')
    parser.add_argument('--dt', type=float, default=0.01, help = 'Discretized SDE time interval between two consecutive steps')
    parser.add_argument('--T', type=int, default=1000, help='Number of discrete noising/denoising timesteps')
    parser.add_argument('--N_samples', type=int, default=1000, help = 'Number of samples')
    parser.add_argument('--CONSTRAINED', type=str2bool, default=False, help='Constraint flag')

    args = parser.parse_args()
       
    return args
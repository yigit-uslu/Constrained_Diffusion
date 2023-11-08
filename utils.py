import numpy as np

def score_function_log_univariate_normal(X, mu, var):
    return (mu - X) / var


def make_experiment_name(args):
    experiment_name = f'CONSTRAINED_{args.CONSTRAINED}_N_samples_{args.N_samples}_T_{args.T}_dt_{args.dt}_theta_{args.theta}_mu_{args.mu}_sigma_{args.sigma}'
    return experiment_name

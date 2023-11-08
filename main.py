import numpy as np
import scipy
from DiffusionModel import DiffusionModel
from config import make_parser
from plot_utils import plot_DM_trajectories
from utils import make_experiment_name

def make_constraint_fn(low = 0, high = 1, eta = 1e-2, eval_type = 'Dummy interval'):
    if eval_type == 'Dummy interval':
        def constraint_fn(X_t):
            constraint_slack = eta * (np.maximum(np.minimum(X_t, high), low) - X_t)
            return constraint_slack
    else:
        raise NotImplementedError
    return constraint_fn


def test_forward_sampler():
    theta = 1/2
    mu = 0
    sigma = 1
    dt = 0.01
    T = 100
    N_samples = 1000

    # X_0 = np.random.uniform(low=0, high=1, size=(N_samples,))
    X_0 = 0.5 * np.ones((N_samples,))

    DM = DiffusionModel(theta=theta, mu=mu, sigma=sigma, X_0=X_0)
    X_over_time = DM.forward(T=T, dt=dt)

    T = len(X_over_time)
    for tt in [int(x) for x in range(0, T, T // 5)]:
        print('t: ', tt)
        t = X_over_time[tt].t
        X = X_over_time[tt].X
        mu_t, var_t = DM.sample(X_over_time[0].X, t)
        print(f'Conditional distribution at timestep {t} N(\mu, \Var) (anl.) \tMean = {np.mean(mu_t)}, Variance = {var_t}')
        print(f'Conditional distribution at timestep {t} N(\mu, \Var) (sim.) \tMean = {np.mean(X)}, Variance = {np.var(X)}')


def main():
    args = make_parser()
    print(args)
    # theta = 1/2 # scale parameter
    # mu = 0 # drift term
    # sigma = 1. # noise level
    # dt = 0.01 # discretized SDE time interval between two consecutive steps X_t+1 and X_t
    # T = 1000 # number of discrete noising/denoising timesteps
    # N_samples = 1000 # number of samples
    # CONSTRAINED = False

    experiment_name = make_experiment_name(args)

    if args.CONSTRAINED:
        constraint_fn = make_constraint_fn(low=1, high=2, eta=1.0)
    else:
        constraint_fn = None

    # Sample prior
    X_0 = np.random.uniform(low=0, high=5, size=(args.N_samples,))
    # X_0 = 0.5 * np.ones((N_samples,))

    # Instantiate the Diffusion Model
    DM = DiffusionModel(theta=args.theta, mu=args.mu, sigma=args.sigma, X_0=X_0, constraint_fn = constraint_fn)
    X_forward_over_time = DM.forward(T=args.T, dt=args.dt)
    plot_DM_trajectories(X_forward_over_time, n_trajectories = min(100, args.N_samples), save_title = 'Forward diff.', save_path=f'{args.ROOT}/{experiment_name}/figs/forward')

    X_reverse_over_time = DM.reverse(X_T=X_forward_over_time[-1], T=args.T, dt=args.dt)
    plot_DM_trajectories(X_reverse_over_time, n_trajectories = min(100, args.N_samples), ref_dist=scipy.stats.uniform(loc=X_0.min(), scale=X_0.max() - X_0.min()).pdf,
                         save_title = 'Reverse diff.' + args.CONSTRAINED * '(constrained)', save_path=f'{args.ROOT}/{experiment_name}/figs/reverse')


if __name__ == "__main__":
    main()
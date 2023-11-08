import os
from matplotlib import pyplot as plt
import numpy as np
import scipy

plt.rcParams['patch.edgecolor'] = 'none'
# plt.rcParams['figure.figsize'] = (18, 6)
FIG_ROWSIZE = 6
FIG_COLSIZE = 6
FIG_FONTSIZE = 8


def plot_DM_trajectories(X_over_time, t_over_time = None, n_trajectories = None, plot_hist = True, ref_dist = None, save_title = '', save_path = './figs'):

    fig, axs = plt.subplots(1, 1)

    if t_over_time is None:
        all_t = []
        all_X = []
        for X_t in X_over_time:
            all_t.append(X_t.t)
            all_X.append(X_t.X)
        all_t = np.array(all_t)
        all_X = np.stack(all_X, 0) # [T, N_samples, ]
        all_X = all_X.reshape(all_X.shape[0], *all_X.shape[1:])
    else:
        all_t = np.array(all_t)
        all_X = np.array(all_X)

    if n_trajectories is not None:
        subset_idx = np.random.choice(all_X.shape[1], n_trajectories)
    else:
        n_trajectories = all_X.shape[1]

    axs.plot(all_t, all_X[:, subset_idx])
    axs.set_xlabel('Time (s)')
    axs.set_ylabel(r'$X_t$')
    axs.grid(True)
    axs.set_title(f'{n_trajectories} DM trajectories {save_title}')
    os.makedirs(f'{save_path}', exist_ok = True)
    plt.savefig(f'{save_path}/DM_trajectories.png', dpi=300)

    if plot_hist:
        plot_DM_hist(all_X, plot_t = [int(x) for x in np.linspace(0, all_X.shape[0]-1, 6)], ref_dist = ref_dist, save_title=save_title, save_path = save_path)
    plt.close('all')


def plot_DM_hist(all_X, plot_t = [0, -1], ref_dist = None, save_title = '', save_path = './figs'):
    n_rows = 2
    n_cols = int(np.ceil(len(plot_t) / 2))

    fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * FIG_COLSIZE, n_rows * FIG_ROWSIZE))
    binwidth = 0.05
    for i, ax in enumerate(axs.flat):
        bins = np.arange(min(all_X[plot_t[i]].reshape(-1)) - binwidth, max(all_X[plot_t[i]].reshape(-1)) + binwidth, binwidth)
        ax.hist(all_X[plot_t[i]].reshape(-1), label = r'$X_{{{}}}$'.format((plot_t[i] + all_X.shape[0]) % all_X.shape[0]), density = True, bins = bins, alpha = 0.5)
        if ref_dist is not None:
            ax.plot(bins, ref_dist(bins), '-r', label = r'$X_{\infty}$')
        else:
            ax.plot(bins, scipy.stats.norm.pdf(bins), '-r', label = r'$\mathcal{N}(0, 1)$')
        ax.grid(True)
        ax.legend(fancybox = True)
    # plt.show()
    fig.suptitle(f'Histograms {save_title}')
    os.makedirs(f'{save_path}', exist_ok = True)
    plt.savefig(f'{save_path}/DM_hist.png', dpi=300)

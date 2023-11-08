import numpy as np
from collections import namedtuple
import torch
import scipy
import tqdm

from utils import score_function_log_univariate_normal

diffusion_state = namedtuple("diffusion_state", ['t', 'X'])


class Prior():
    '''
    TO DO: Add a prior sampling method to the Diffusion Model
    '''
    def __init__(self, params):
        self.dist = params.dist
        self.loc = params.loc
        self.scale = params.scale

        if self.dist == 'uniform':
            self.dist = scipy.stats.uniform
        elif self.dist == 'normal':
            self.dist = scipy.stats.normal
        else:
            raise NotImplementedError

    def sample(self, N_samples):
        return self.dist.rvs(N_samples)



class DiffusionModel():
    '''
    Discretized Ornstein-Uhlenbeck process $dX_t = \Theta (\mu - X_t)dt + \sigma dW_t$
    '''
    def __init__(self, theta = 1/2, mu = 0, sigma = 1., X_0 = None, X_T = None, constraint_fn = None, prior = 'uniform'):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.X_0 = X_0 # p_data
        self.forward_state = diffusion_state(t=0, X=X_0)
        self.reverse_state = diffusion_state(t=0, X=X_T)
        self.score_model = None
        self.constraint_fn = constraint_fn

    def sample(self, X_0, t, X_t = None):
        mu_t = X_0 * np.exp(-self.theta * t) + self.mu * (1 - np.exp(-self.theta * t)) # ndarray (N_samples,)
        var_t = self.sigma**2 / (2 * self.theta) * (1 - np.exp(-2 * self.theta * t)) # float

        q_t = None
        if X_t is not None:
            q_t = np.stack([scipy.stats.norm.pdf(X_t, loc = mu, scale = var_t) for mu in mu_t.tolist()], axis=0) # [n_samples, n_samples] each row is q_t(X) cond. on  different X_0

        return mu_t, var_t, q_t
        
    def estimate_score_function(self, X_t):
        mu_t, var_t, q_t = self.sample(X_0 = self.X_0.X, t=X_t.t, X_t=X_t.X)
        log_scores = score_function_log_univariate_normal(np.repeat(np.expand_dims(X_t.X, 0), mu_t.shape[0], axis = 0), mu=mu_t, var=var_t) # [n_samples, n_samples] each row is samples from log q(X_t | X_0 = x_0) for a fixed x_0
        score = np.mean(log_scores * q_t, 0) / np.mean(q_t, 0)
        # score = np.mean(scores, 0) / (1 + 0 * np.mean(q_t, 0))
        # score = np.float32(score)
        # score = - (X_t.X - mu_t) / var_t
        return score
        

    def step(self, X_t, dt):
        '''
        X_t is a tuple of time and X values, dt is negative for reverse process
        '''
        if self.constraint_fn is None:
            dX_t = self.theta * (self.mu - X_t.X) * np.abs(dt) + self.sigma * np.sqrt(np.abs(dt)) * np.random.randn(*X_t.X.shape)
            dX_t = dX_t - dt * self.sigma**2 * (np.sign(dt) == -1) * np.nan_to_num(self.estimate_score_function(X_t), 0)
        else:
            dX_t = self.theta * (self.mu - X_t.X) * np.abs(dt) + self.sigma * np.sqrt(np.abs(dt)) * np.random.randn(*X_t.X.shape)
            dX_t = dX_t - dt * (np.sign(dt) == -1) * (self.sigma**2 * np.nan_to_num(self.estimate_score_function(X_t), 0) + self.constraint_fn(X_t.X))
            
        return diffusion_state(t=X_t.t + dt, X = X_t.X + dX_t)
        

    def forward(self, X_0 = None, T = 100, dt = 0.01):
        X = X_0 if X_0 is not None else self.forward_state
        self.X_0 = X # p_data
        X_over_time = [X]
        for t in tqdm.tqdm(range(T)):
            X = self.step(X, dt=dt)
            self.forward_state = X
            X_over_time.append(X)
        self.reverse_state = diffusion_state(t = X_over_time[-1].t, X = np.random.randn(*X_over_time[-1].X.shape))
        return X_over_time
    

    def reverse(self, X_T = None, T = 100, dt = 0.01):
        X = X_T if X_T is not None else self.reverse_state
        X_over_time = [X]
        for t in tqdm.tqdm(range(T)):
            X = self.step(X, dt=-dt)
            self.reverse_state = X
            X_over_time.append(X)

        return X_over_time

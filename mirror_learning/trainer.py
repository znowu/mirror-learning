import numpy as np
import torch
from mirror_learning import envs, neihbourhoods, drifts
from mirror_learning.agents import *
import torch.random
np.random.seed(0)
torch.manual_seed(0)

class Training:

    def __init__(self, config):

        envs = config["envs"]
        drifts = config["drifts"]
        neighbourhoods = config["neighbourhoods"]
        samplings = config["samplings"]

        self.n_threads = len(envs)
        self.envs = envs
        self.agents = [Agent(envs[i], neighbourhoods[i], drifts[i], samplings[i])
                        for i in range(self.n_threads)]

    def get_mean_v(self, env, agent):
        fitted_v = env.fit_v(soft_to_direct(agent.pi.detach()))
        mean_v, n_states = 0., 0
        for state in env.states:
            if not hasattr(env, "barrier") or state not in env.barrier:
                mean_v += fitted_v[state]
                n_states += 1
        mean_v /= n_states
        return mean_v

    def train(self, n_iters=20):

        min_drift_val = np.zeros((n_iters+1, self.n_threads))
        exp_drift_val = np.zeros((n_iters+1, self.n_threads))
        V = np.zeros((n_iters+1, self.n_threads))
        V[0, :] = np.array([
            self.get_mean_v(self.envs[th], self.agents[th]) for th in range(self.n_threads)
            ])

        for i in range(n_iters):

            Q = [self.envs[th].fit_q(soft_to_direct(self.agents[th].pi.detach()))
                for th in range(self.n_threads) ]
            _ = list(zip(*[self.agents[th].mirror_step(Q[th]) for th in range(self.n_threads)]))

            min_drift, exp_drift = _[0], _[1]
            min_drift_val[i+1, :] = min_drift_val[i, :] + np.array(min_drift)
            exp_drift_val[i+1, :] = exp_drift_val[i, :] + np.array(exp_drift)

            V[i+1, :] = np.array([
                self.envs[th].fit_v(soft_to_direct(self.agents[th].pi.detach())).mean().item()
                for th in range(self.n_threads) ])

        return {"V": V, "min_drift": min_drift_val, "exp_drift": exp_drift_val}


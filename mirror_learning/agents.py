import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from drifts import *
np.random.seed(0)
torch.manual_seed(0)

def normalize(rows):
    rows = rows/torch.sum(rows, dim=1)[:, np.newaxis]
    return rows

def to_canonical(pi):
    return pi[:, :-1]

def random_soft(n_states, n_actions):
    return torch.rand((n_states, n_actions-1))

def random_canonical(n_states, n_actions):
    rows = torch.rand((n_states, n_actions - 1))
    rows = (n_actions - 1)*normalize(rows)/n_actions
    return rows

def canonical_to_direct(pi):
    return torch.cat([pi, 1-torch.sum(pi, dim=-1).view(-1, 1)], dim=-1)

def soft_to_direct(pi):
    all_params = torch.cat([pi, torch.ones(pi.shape[0], 1)], dim=-1)
    exps = torch.exp(all_params)
    return exps/torch.sum(exps, dim=-1).view(pi.shape[0], 1)

def is_valid_canonical(vecs):
    return torch.all(vecs >= 0) and torch.all(torch.sum(vecs, dim=-1) < 1.)

def advantage(Q, pi):
    V = torch.sum(Q*pi, dim=-1)
    return Q-V[:, np.newaxis]

class Agent:

    def __init__(self, env, neighbourhood, drift, sampling, lr=1e-2):
        """
        :param env: Env object, the MDP.
        :param neighbourhood: neighbourhood function.
        :param drift: mirror function.
        :param sampling: function of policy generating sampling state distributions.
        :param lr: learning rate.
        """

        self.n_states = len(env.states)
        self.n_actions = len(env.actions)
        torch.manual_seed(0)
        self.pi = random_soft(self.n_states, self.n_actions)
        self.N = neighbourhood
        self.D = drift
        self.beta = sampling
        self.lr = lr
        self.mini_epochs=1000
        #self.optimizer = optim.Adam(variable(self.pi), lr=self.lr)


    def act(self, state):
        """
        Sample an action from the policy.
        """
        return np.random.choice(self.n_actions, 1, soft_to_direct(self.pi[state, :]) )

    def loss(self, Q, pi_base, pi_bar):
        """
        Compute the loss, given the current policy.
        """
        beta = self.beta(pi_base)
        mirror = expected_drift(self.D, beta)
        pi_bar_dir = soft_to_direct(pi_bar)
        A = advantage(Q, pi_base)
        objective = torch.sum(A*pi_bar_dir, dim=-1).long()
        mirror_value = mirror(pi_base, pi_bar_dir).long()
        return -(torch.dot(beta, objective) - mirror_value)

    def mirror_step(self, Q):

        pi_base = soft_to_direct(self.pi.detach())
        pi_bar = Variable(self.pi.data, requires_grad=True)

        for i in range(self.mini_epochs):

            # compute the drift value
            beta = self.beta(pi_base)
            pi_bar_dir = soft_to_direct(pi_bar)
            drift_value = expected_drift(self.D, self.beta)(pi_base, pi_bar_dir).float()

            # compute the objective value
            A = advantage(Q, pi_base)
            objective = torch.sum(A * pi_bar_dir, dim=-1).float()

            # combine into loss
            loss = -(torch.dot(beta, objective) - drift_value)
            loss.backward()
            candidate_pi = pi_bar.data - (self.lr * pi_bar.grad.data)
            coef = 0.9
            while not self.N(pi_base, soft_to_direct(candidate_pi).detach()):
                candidate_pi = pi_bar.data - coef * self.lr * pi_bar.grad.data
                coef *= 0.9

            pi_bar.grad.data.zero_()
            pi_bar.data = candidate_pi

        self.pi = pi_bar.data
        return min_drift(self.D)(pi_base, soft_to_direct(self.pi)), \
               expected_drift(self.D, self.beta(self.pi), functional=False)(pi_base, soft_to_direct(self.pi))





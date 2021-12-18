import numpy as np
import torch
import copy
np.random.seed(0)
torch.manual_seed(0)
class Env:
    """
    An abstract class of an MDP that provides a template of interaction
    state > action > reward > next state.
    """
    def __init__(self, states, actions, terminal_states):
        """
        :param states: state space
        :param actions: action space
        :param terminal_states
        """
        self.states = states
        self.actions = actions
        self.terminal_states = terminal_states
        self.x = None
        self.done = False
        self.gamma = None
        self.reward_matrix_value = None

    def initialize(self):
        """
        :return: how do you start an experience in the MDP.
        """
        pass

    def reward(self, state, action):
        """
        :return: the reward function.
        """
        pass

    def reward_matrix(self):
        if self.reward_matrix_value is None:
            self.reward_matrix_value =\
                torch.tensor([[self.reward(state, action) for action in self.actions] for state in self.states])
        return self.reward_matrix_value


    def transit(self, state, action):
        """
        :return: output the next state conditioned on
        the current state and action.
        """
        pass

    def play(self, action):
        """
        :param action: the action taken by the agent
        :return: the next state, and the reward. update
        the status of the experience (finished or not).
        """
        reward = self.reward(self.x, action)
        next_state = self.transit(self.x, action)
        if next_state in self.terminal_states:
            self.done = True
        return reward, next_state

    def fit_v(self, pi):
        """
        Fit value function of the new policy.
        """
        R = self.reward_matrix()
        R_pi = torch.sum(R*pi, dim=-1)
        next_state = torch.tensor([ [self.transit(state, action) for action in self.actions] for state in self.states])
        V = np.zeros( len(self.states) )

        for i in range(200):
            next_V = torch.tensor([ [0 if next_state[i, j] in self.terminal_states
                                     else V[next_state[i, j]] for j in range(next_state.shape[1])]
                                    for i in range(next_state.shape[0]) ])
            next_V = torch.sum(next_V*pi, dim=-1)
            V = R_pi + self.gamma*next_V

        return V

    def fit_q(self, pi):
        """
        Fit the state-action value function of the new policy.
        """
        V = self.fit_v(pi)
        R = self.reward_matrix()
        Q = torch.tensor([
            [ R[state, action] if self.transit(state, action) in self.terminal_states
                else R[state, action] + self.gamma*V[self.transit(state, action)]
              for action in self.actions] for state in self.states])
        return Q






class Bandit(Env):
    """
    Multi-Armed Bandit. One state, tabular reward.
    """
    def __init__(self, n_arms, rewards):
        states = [0]
        terminal_states = [1]
        actions = np.arange(n_arms)
        super().__init__(states, actions, terminal_states)
        self.rewards = rewards
        self.gamma = 0

    def transit(self, state, action):
        return 1

    def initialize(self):
        self.done = False
        self.x = 0

    def reward(self, state, action):
        return self.rewards[action]


class TabularGame(Env):
    """
    A game with many states, each of which is a multi-armed bandit.
    The goal is to learn to ignore small rewards leading to
    damaging left terminal state, and reach the right terminal
    state with the optimal reward.
    """
    def __init__(self, n_tables, n_actions):
        states = list(range(n_tables))
        actions = list(range(n_actions))
        terminal_states = [-1, n_tables]
        super().__init__(states, actions, terminal_states)
        self.gamma = 0.999

    def initialize(self):
        self.done = False
        self.x = np.random.randint(0, len(self.states)-1)

    def reward(self, state, action):

        if state in self.terminal_states:
            return 0.

        elif state ==0 and action ==0:
            return -10

        elif state == len(self.states)-1 and action == len(self.actions)-1:
            return 10

        else:
            if action == 0:
                return 0.1
            elif action == len(self.actions)-1:
                return -0.1
            return 0

    def transit(self, state, action):

        if state in self.terminal_states:
            return state

        elif action == 0:
            return state-1

        elif action == len(self.actions)-1:
            return state+1

        return state


class GridWorld(TabularGame):

    def __init__(self, n_rows = 5, n_cols = 5):

        super().__init__(n_tables= n_rows*n_cols - 2, n_actions=4)
        self.goal = n_rows*n_cols
        self.bomb = -1
        self.barrier = [(n_rows-2)*n_cols + i for i in range(n_cols-1)]
        self.n_rows = n_rows
        self.n_cols = n_cols

    def initialize(self):
        self.done = False
        n_s = len(self.states)
        x = np.random.randint(0, n_s)
        while x in self.barrier:
            x = np.random.randint(0, n_s)

        self.x = x

    def reward(self, state, action):

        if self.transit(state, action) == self.bomb:
            return -100

        return -1.

    def transit(self, state, action):
        """
        actions
        0 -> go right
        1 -> go up
        2 -> go left
        3 -> go down
        """
        if action == 0:
            if state+2 % self.n_cols == 0 or state + 1 in self.barrier:
                return state
            else:
                return state + 1

        if action == 1:
            if state + self.n_rows > len(self.states) or state + self.n_rows in self.barrier:
                return state
            else:
                return state + self.n_rows

        if action == 2:
            if state + 1 % self.n_cols == 0 or state - 1 in self.barrier:
                return state
            else:
                return state-1

        else:
            if state - self.n_rows < -1 or state - self.n_rows in self.barrier:
                return state
            else:
                return state - self.n_rows






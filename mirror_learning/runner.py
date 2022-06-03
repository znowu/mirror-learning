import numpy as np
import torch
from trainer import *
from config import bandit_config, tabular_config, gridworld_config
import matplotlib.pyplot as plt
np.random.seed(0)
torch.manual_seed(0)



def main():

    plt.style.use('seaborn-whitegrid')

    experiments = [("Single-Step", bandit_config, 50), ("Tabular", tabular_config, 50),
                   ("GridWorld", gridworld_config, 50)]

    for name, config, n_iters in experiments:
        training = Training(config)
        summary = training.train(n_iters)
        v = summary["V"]
        drift = summary["exp_drift"]

        fig = plt.figure()

        for i in range(len(config["envs"])):
            plt.plot(5*np.arange(0, 11), v[::5, i], marker='v', label=config["drift_names"][i], c=config["colors"][i])
            plt.plot(5*np.arange(0, 11), drift[::5, i]+v[0, i], ":", marker='o',  c=config["colors"][i])

        print(v[-1, :])

        plt.title("{} with Different Drifts and KL-Neighbourhood".format(name))
        plt.xlabel("Iteration")
        plt.ylabel("Average Return")
        plt.legend()
        fig.savefig("{} experiments KL".format(name))


if __name__ == "__main__":
    main()


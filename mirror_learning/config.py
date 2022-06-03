import numpy as np
import torch
import envs, drifts, neihbourhoods

"""
Here, configurations for the training can be specified.
"""

bandit_config = {
    "envs": 4* [ envs.Bandit(5, np.array([10., 0, 1., 0., 5.])) ],
    "drifts": [drifts.Zero(), drifts.TVSq(50), drifts.EuclidSq(50), drifts.KL(50)],
    "drift_names": ["Zero", "TV-squared", "L2-squared", "KL"],
    "neighbourhoods": [
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(1), functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(1), functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(1), functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(1), functional=False), 0.05),
    ],
    "samplings": 4*[lambda pi: torch.ones(1)],
    "colors": ["tab:blue", "tab:purple", "tab:cyan", "tab:red"]
}

tabular_config = {
    "envs": 4* [ envs.TabularGame(5, 3) ],
    "drifts": [drifts.Zero(), drifts.TVSq(50), drifts.EuclidSq(50), drifts.KL(50)],
    "drift_names": ["Zero", "TV-squared", "L2-squared", "KL"],
    "neighbourhoods": [
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(5)/5, functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(5)/5, functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(5)/5, functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(5)/5, functional=False), 0.05),
    ],
    "samplings": 4*[lambda pi: torch.ones(5)/5],
    "colors": ["tab:blue", "tab:purple", "tab:cyan", "tab:red"]
}


gridworld_config = {
    "envs" : 4* [envs.GridWorld(5, 5)],
    "drifts": [drifts.Zero(), drifts.TVSq(50), drifts.EuclidSq(50), drifts.KL(50)],
    "drift_names": ["Zero", "TV-squared", "L2-squared", "KL"],
    "neighbourhoods": [
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(23)/23, functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(23)/23, functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(23)/23, functional=False), 0.05),
        neihbourhoods.constraint_neighbourhood(
            drifts.expected_drift(drifts.KL(1), torch.ones(23)/23, functional=False), 0.05),
    ],
    "samplings": 4*[lambda pi: torch.ones(23)/23],
    "colors": ["tab:blue", "tab:purple", "tab:cyan", "tab:red"]
}
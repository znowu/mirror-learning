import numpy as np
from drifts import *

def trivial_neighbourhood():
    return lambda pi1, pi2: True


def constraint_neighbourhood(constraint_function, delta):
    return lambda pi1, pi2: (constraint_function(pi1, pi2) <= delta)


import random

import numpy as np


def reseed(seed=423652346):
    random.seed(seed)
    np.random.seed(seed)

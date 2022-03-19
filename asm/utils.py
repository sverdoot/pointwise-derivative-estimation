import random

import numpy as np


def random_seed(seed: int):
    """Fix random seed."""
    np.random.seed(seed)
    random.seed(seed)

import numpy as np


class Individual:
    def __init__(self, x0: np.ndarray):
        self.x: np.ndarray = x0
        self.rank: int = 0
        self.f_best: np.ndarray
        self.x_best: np.ndarray

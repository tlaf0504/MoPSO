import numpy as np
from abc import abstractmethod


class Problem:
    def __init__(self, n_var: int, n_obj:int, xl: np.ndarray, xu: np.ndarray):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        """ Method overwritten by the user to implement the problem evaluation
        """
        pass

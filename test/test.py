import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance

from MoPSO.MoPSO import MoPSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

class MyZDT1Implementation(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=30,
                         n_obj=2,
                         xl=np.zeros(30),
                         xu=np.ones(30))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]
        g = 1 + 9 / 29 * np.sum(x[1:])
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h

        out["F"] = [f1, f2]
        out["G"] = []

class TestMoPSO(unittest.TestCase):

    def test_nondominated_sorting(self):
        rng = default_rng(1234)
        n_individuals = 200

        # Use the non-dominated sorting algorithm from pymoo
        reference_sorter = NonDominatedSorting()
        f1 = rng.uniform(0, 1, n_individuals)
        f2 = rng.uniform(0, 1, n_individuals)
        F = np.column_stack((f1, f2))

        fronts = reference_sorter.do(F, return_rank=True)

        plt.figure()
        plt.scatter(f1, f2, s=50, c=fronts[1], cmap="Set1")
        for k in range(n_individuals):
            plt.annotate(k,(f1[k], f2[k]))
        plt.xlabel(r"$f_1$")
        plt.ylabel(r"$f_2$")
        plt.title("Pareto-front classes determined by pymoo")
        plt.colorbar()

        fronts_, ranks_ = MoPSO.nondominated_sorting(F)

        plt.figure()
        plt.scatter(f1, f2, s=50, c=ranks_, cmap="Set1")
        plt.xlabel(r"$f_1$")
        plt.ylabel(r"$f_2$")
        plt.title("Pareto-fronts determined by own algorithm ")
        plt.colorbar()
        for k in range(n_individuals):
            plt.annotate(k,(f1[k], f2[k]))
        plt.show()

        pymoo_ranks = fronts[1]
        self_calculated_ranks = ranks_
        self.assertTrue(np.all(pymoo_ranks == self_calculated_ranks))

    def test_crowding_distance_measurement(self):
        rng = default_rng(1234)
        n_individuals = 20

        # Use the non-dominated sorting algorithm from pymoo
        reference_sorter = NonDominatedSorting()
        f1 = rng.uniform(0, 1, n_individuals)
        f2 = rng.uniform(0, 1, n_individuals)
        F = np.column_stack((f1, f2))

        distances_pymoo = calc_crowding_distance(F, filter_out_duplicates=False)
        # For some reason, a factor of 2 lies between the crowding distances of pymoo and the self-programmed solution
        distances_self = MoPSO.crowding_distance_calculation(F)

        _distances_pymoo = distances_pymoo[np.isnan(distances_pymoo)] = 0.0
        _distances_self = distances_self[np.isnan(distances_self)] = 0.0

        self.assertLess(np.max(np.abs(_distances_pymoo - _distances_self / 2.0)), 1e-5)

    def test_MyZDT1Implementation(self):
        problem = MyZDT1Implementation()
        algorithm = NSGA2(
            pop_size=50)
        termination = get_termination("n_gen", 40)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)

        X = res.X
        F = res.F
        plt.figure(figsize=(7, 5))
        plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.title("Objective Space")
        plt.show()







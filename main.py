import os
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from numpy.random import default_rng

from MoPSO.MoPSO import MoPSO
from MoPSO.Problem import Problem
import time
import datetime
import glob
import shutil
import re


class ZDT1(Problem):
    def __init__(self):
        super(ZDT1, self).__init__(n_var=30,
                                   n_obj=2,
                                   xl=np.zeros(30),
                                   xu=np.ones(30))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9/29 * np.sum(x[:, 1:], axis=1)
        h = 1 - np.sqrt(f1/g)
        f2 = g*h

        assert not (np.any(np.isnan(f1)) or np.any(np.isnan(f2)))
        out["objectives"] = np.column_stack((f1, f2))

class ZDT2(Problem):
    def __init__(self):
        super(ZDT2, self).__init__(n_var=30,
                                   n_obj=2,
                                   xl=np.zeros(30),
                                   xu=np.ones(30))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9/29 * np.sum(x[:, 1:], axis=1)
        h = 1 - np.power(f1/g, 2)
        f2 = g*h

        assert not (np.any(np.isnan(f1)) or np.any(np.isnan(f2)))
        out["objectives"] = np.column_stack((f1, f2))

class RAND(Problem):
    def __init__(self):
        super(RAND, self).__init__(n_var=2,
                                   n_obj=2,
                                   xl=np.zeros(2),
                                   xu=np.ones(2))
    def _evaluate(self, x, out, *args, **kwargs):
        rng = default_rng(1234)
        n_individuals = x.shape[0]

        f1 = rng.uniform(0, 1, n_individuals)
        f2 = rng.uniform(0, 1, n_individuals)

        out["objectives"] = np.column_stack((f1, f2))


def callback_video(individuals, objective_values, destination_directory):

    names = os.listdir(destination_directory)
    if len(names) == 0:
        idx = 1
    else:
        indices_str = list(map(lambda x: int(re.search("[0-9]+", x).group(0)), names))
        indices = list(map(lambda x: int(x), indices_str))
        idx = np.max(indices) + 1

    f1 = objective_values[:, 0]
    f2 = objective_values[:, 1]

    n_individuals = len(individuals)
    ranks = np.zeros(n_individuals, dtype=int)
    for k in range(n_individuals):
        ranks[k] = individuals[k].rank

    fig = plt.figure()
    plt.scatter(f1, f2, s=25, c=ranks, cmap="Set1")
    plt.xlabel(r"$f_1$")
    plt.ylabel(r"$f_2$")
    plt.title("Function-space, cycle {:d}".format(idx))
    #plt.xlim((0, 1))
    #plt.ylim((0, 1))
    plt.colorbar()

    fig.savefig(os.path.join(destination_directory, "frame_{:d}.png".format(idx)))
    plt.close()

def callback(individuals, objective_values, label_individuals:bool=True):
    f1 = objective_values[:, 0]
    f2 = objective_values[:, 1]

    n_individuals = len(individuals)
    ranks = np.zeros(n_individuals, dtype=int)
    for k in range(n_individuals):
        ranks[k] = individuals[k].rank

    plt.figure()
    plt.scatter(f1, f2, s=25, c=ranks, cmap="Set1")
    plt.xlabel(r"$f_1$")
    plt.ylabel(r"$f_2$")
    plt.title("Function-space")
    plt.colorbar()
    if label_individuals:
        for k in range(n_individuals):
            plt.annotate(k, (f1[k], f2[k]))
    plt.show()


def main():
    update_strategy = "strategy_1"
    boundary_strategy = "strategy_1"

    res_subdir = "boundary_{:s}__update_{:s}".format(boundary_strategy, update_strategy)
    if os.path.isdir(res_subdir):
        shutil.rmtree(res_subdir)
    os.mkdir(res_subdir)
    
    problem = ZDT1()
    optimizer = MoPSO(problem)
    optimizer.run(n_individuals=100,
                  n_cycles=100,
                  callback=lambda a, b: callback_video(a, b, res_subdir),
                  boundary_strategy=boundary_strategy,
                  update_strategy=update_strategy)

    callback(optimizer.individuals, optimizer.objective_values, label_individuals=False)

    #problem = RAND()
    #optimizer = MoPSO(problem)
    #optimizer.run(n_individuals=25, n_cycles=100, callback=callback)






if __name__ == '__main__':
    main()


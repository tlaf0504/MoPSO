import numpy as np
from numpy.random import default_rng
from NDS_crowd_gauss import NDS_crowd_gauss

class update_strategy_2:
    def __init__(self):
        pass
    @staticmethod
    def update_population(ranks, fronts, objective_values, population):
        n_individuals = len(ranks)
        n_optimization_variables = len(population[0].x)

        NDS_position_updates = np.zeros((n_individuals, n_optimization_variables))

        # Get position updates along rank 0 individual
        rank_0_front = np.array(fronts[0])
        for k in range(n_individuals):
            rank_of_current_individual = ranks[k]
            if rank_of_current_individual >= 1:
                NDS_position_updates[k, :] = \
                    NDS_crowd_gauss.calculate_rank_0_velocity_component(k,
                                                                        rank_0_front, population, objective_values)


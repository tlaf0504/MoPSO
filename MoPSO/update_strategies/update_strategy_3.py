import numpy as np
from numpy.random import default_rng
from MoPSO.update_strategies.update_strategy_2 import update_strategy_2

class update_strategy_3:
    def __init__(self):
        pass
    @staticmethod
    def update_population(ranks, fronts, objective_values, population):
        n_individuals = len(ranks)
        n_optimization_variables = len(population[0].x)

        NDS_position_updates = np.zeros((n_individuals, n_optimization_variables))
        crowd_position_updates = np.zeros((n_individuals, n_optimization_variables))

        # Get position updates along rank 0 individual
        rank_0_front = np.array(fronts[0])
        for k in range(n_individuals):
            rank_of_current_individual = ranks[k]
            if rank_of_current_individual >= 1:
                NDS_position_updates[k, :] = \
                    update_strategy_2.calculate_rank_0_velocity_component(k,
                                                                          rank_0_front, population, objective_values)

        # Get position updates along personal and global best objective values
        global_best_position_updates, personal_best_position_updates = \
            update_strategy_3.calculate_objective_minimum_velocity_components(population,
                                                                            objective_values,
                                                                            global_best_weight=0.1,
                                                                            personal_best_weight=0.1)

        for k in range(n_individuals):
            population[k].x = population[k].x + NDS_position_updates[k, :] + \
                              global_best_position_updates[k, :] + personal_best_position_updates[k, :]

    @staticmethod
    def calculate_objective_minimum_velocity_components(
            population: list,
            objective_values,
            personal_best_weight: float = 1.0,
            global_best_weight: float = 1.0,
            rng=None):

        n_objectives = objective_values.shape[1]
        n_optimization_variables = population[0].x.shape[0]
        n_individuals = len(population)
        global_best_position_updates = np.zeros((n_individuals, n_optimization_variables))
        personal_best_position_updates = np.zeros((n_individuals, n_optimization_variables))

        if rng is None:
            rng = default_rng()

        global_best_individuals_idx = np.argmin(objective_values, axis=0)
        for k in range(n_individuals):
            current_individual = population[k]

            # Indices of objectives to choose for personal and global best
            objective_selectors = rng.integers(0, n_objectives, 2)
            personal_best_objective_idx = objective_selectors[0]
            global_best_objective_idx = objective_selectors[1]

            # Distance to personal best for selected objective
            personal_best_distance = rng.uniform(0, 1) * personal_best_weight * \
                                     (current_individual.x_best[:, personal_best_objective_idx] - current_individual.x)

            # Distance to global best for selected objective
            global_best_individual = population[global_best_individuals_idx[global_best_objective_idx]]
            global_best_distance = rng.uniform(0, 1) * global_best_weight * \
                                   (global_best_individual.x - current_individual.x)

            global_best_position_updates[k, :] = global_best_distance
            personal_best_position_updates[k, :] = personal_best_distance

        return global_best_position_updates, personal_best_position_updates


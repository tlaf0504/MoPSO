
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from MoPSO.Individual import Individual


class update_strategy_2:
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

            crowd_position_updates[k, :] = \
                update_strategy_2.calculate_crowding_velocity_component(k,
                                                                        np.array(fronts[rank_of_current_individual]),
                                                                        population,
                                                                        objective_values)

        for k in range(n_individuals):
            population[k].x = population[k].x + NDS_position_updates[k, :] + crowd_position_updates[k, :]

    @staticmethod
    def calculate_rank_0_velocity_component(current_individual_idx: int,
                                     other_individuals_idx: np.ndarray,
                                     population: list,
                                     objective_values: np.ndarray,
                                     rng=None,
                                     mu_scale=0.5,
                                     sigma_scale=0.5):
        n_lower_rank_individuals = len(other_individuals_idx)
        if rng is None:
            rng = default_rng()

        # Calculate objectives space distances to other given individuals
        objective_space_distances = np.zeros(n_lower_rank_individuals)
        for k in range(n_lower_rank_individuals):
            # Calculate distance in objective space
            objective_space_distances[k] = np.linalg.norm(objective_values[current_individual_idx, :] -
                                                          objective_values[other_individuals_idx[k], :])

        # Find closest individual
        closest_individual_idx = np.argmin(objective_space_distances)
        closest_individual = other_individuals_idx[closest_individual_idx]

        # Unit-vector and distance to closest individual
        delta_x = population[current_individual_idx].x - population[closest_individual].x
        delta_x_norm = np.linalg.norm(delta_x)
        e_delta_x = delta_x / delta_x_norm

        # Update position
        mu = delta_x_norm * mu_scale
        sigma = delta_x_norm * sigma_scale
        x_update = rng.normal(loc=mu, scale=sigma) * e_delta_x
        if np.any(np.isnan(x_update)):
            print("Nan")
        return x_update

    @staticmethod
    def calculate_crowding_velocity_component(current_individual_idx: int,
                                     individuals_on_front: np.ndarray,
                                     population: list,
                                     objective_values: np.ndarray,
                                     rng=None):

        other_individuals_on_front = individuals_on_front[individuals_on_front != current_individual_idx]

        n_other_individuals_on_front = len(other_individuals_on_front)
        if n_other_individuals_on_front < 2:
            return np.zeros(population[0].x.shape)

        if rng is None:
            rng = default_rng()

        # Calculate objectives space distances to other given individuals
        objective_space_distances = np.zeros(n_other_individuals_on_front)
        for k in range(n_other_individuals_on_front):
            # Calculate distance in objective space
            objective_space_distances[k] = np.linalg.norm(objective_values[current_individual_idx, :] -
                                                          objective_values[other_individuals_on_front[k], :])

        sorted_objective_space_distances_idx = np.argsort(objective_space_distances)
        closest_individuals = other_individuals_on_front[sorted_objective_space_distances_idx]

        individual_i = population[current_individual_idx]
        individual_j = population[closest_individuals[0]] # Closest individual
        individual_k = population[closest_individuals[1]] # Second closest individual

        fi = objective_values[current_individual_idx]
        fj = objective_values[closest_individuals[0]]
        fk = objective_values[closest_individuals[1]]

        delta_fij = np.linalg.norm(fj - fi)
        delta_fik = np.linalg.norm(fk - fi)
        delta_fjk = np.linalg.norm(fj - fk)

        # Current element is boundary element
        if delta_fik > delta_fjk:
            s = rng.normal(loc=1, scale=1)
            delta_xi = -s * (individual_j.x - individual_i.x)
        else:
            p1 = delta_fik / (delta_fij + delta_fik)
            p2 = delta_fij / (delta_fij + delta_fik)

            s = rng.normal(loc=1 / 2, scale=1 / 6)
            delta_p = s - p1

            delta_xi = delta_p * (individual_k.x - individual_i.x) - delta_p * (individual_j.x - individual_i.x)
        return delta_xi








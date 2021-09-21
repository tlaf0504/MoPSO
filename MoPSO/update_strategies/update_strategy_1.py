import numpy as np
from numpy.random import default_rng

class update_strategy_1:
    def __init__(self):
        pass

    @staticmethod
    def update_population(ranks, fronts, crowding_distances, individuals, rng=None):
        n_var = individuals[0].x.shape[0]
        n_individuals = len(individuals)
        n_fronts = len(fronts)  # Number of pareto-fronts
        n_rank_1_individuals = len(fronts[0])  # Number of rank 1 individuals

        if rng is None:
            rng = default_rng()

        # Sort individuals on each front in ascending order according to their crowding distances
        idx_sorted_crowding_distances = []
        for k in range(n_fronts):
            idx_sorted_crowding_distances.append(np.argsort(crowding_distances[k]))

        # Calculate new particle-positions
        new_positions = np.zeros((n_individuals, n_var))
        for k in range(n_individuals):
            current_individual = individuals[k]
            rank_of_current_individual = ranks[k]

            # Select a random rank 1 individual
            idx_rank_1_individual = rng.integers(low=0, high=n_rank_1_individuals, size=1)[0]
            rank_1_individual = fronts[0][idx_rank_1_individual]

            # The the individual with the highest crowding distance on the current front.
            individual_with_highest_crowding_distance = idx_sorted_crowding_distances[rank_of_current_individual][-1]

            delta_x_NDS = individuals[rank_1_individual].x - current_individual.x
            delta_x_crowd = individuals[individual_with_highest_crowding_distance].x - current_individual.x

            weight_NDS = rng.uniform(0, 1.2, 1)
            weight_crowd = rng.uniform(0, 2, 1)
            new_positions[k] = current_individual.x + weight_NDS * delta_x_NDS + weight_crowd * delta_x_crowd

        return new_positions
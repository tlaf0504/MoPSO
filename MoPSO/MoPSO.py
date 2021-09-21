import numpy as np
from abc import abstractmethod

from numpy.random import default_rng

from MoPSO.Individual import Individual
from MoPSO.Problem import Problem
from MoPSO.update_strategies.update_strategy_1 import update_strategy_1
from MoPSO.update_strategies.update_strategy_2 import update_strategy_2
from MoPSO.update_strategies.update_strategy_3 import update_strategy_3

class MoPSO:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj

        self.n_individuals = 0
        self.individuals = []
        self.objective_values: np.ndarray = np.array([])
        self.rng = None
        self.callback: callable = None

    def create_initial_population(self, n_individuals: int):

        # Clear individuals from previous run
        self.individuals.clear()

        # Create a uniformly distributed population in the parameter-space hypercube defined by the upper and lower
        # parameter-bounds xu and xl
        rng = default_rng(1234)

        # Sample from unit hypercube
        samples = rng.uniform(0, 1, n_individuals * self.n_var)
        samples = samples.reshape((n_individuals, self.n_var))

        # Scale to parameter hypercube
        samples = self.problem.xl + samples * (self.problem.xu - self.problem.xl)

        for k in range(n_individuals):
            self.individuals.append(Individual(samples[k, :]))
            self.individuals[k].f_best = np.inf * np.ones(self.n_obj)
            self.individuals[k].x_best = np.zeros((self.n_var, self.n_obj))

    def evaluate(self, *args, **kwargs):
        """ Evaluate the objectives and constraints of the current population
        """
        out = {"objectives": np.array([])}
        x_ = np.array(list(map(lambda foo: foo.x, self.individuals)))
        self.problem._evaluate(x_, out, *args, **kwargs)
        self.objective_values = out["objectives"]

    def update_population(self, update_strategy: str = "strategy_1", boundary_strategy: str = "strategy_1"):
        # Do non-dominated sorting and calculate crowding-distances for each front
        fronts, ranks = self.nondominated_sorting(self.objective_values)


        # Update ranks and personal bests in population
        for k in range(self.n_individuals):
            self.individuals[k].rank = ranks[k]

            # Update personal bests
            objectives_of_current_individual = self.objective_values[k, :]
            idx_best_new = np.where(objectives_of_current_individual < self.individuals[k].f_best)[0]
            self.individuals[k].f_best[idx_best_new] = objectives_of_current_individual[idx_best_new]
            for idx in idx_best_new:
                self.individuals[k].x_best[:, idx] = self.individuals[k].x

        if self.callback is not None:
            self.callback(self.individuals, self.objective_values)

        if update_strategy == "strategy_1":
            crowding_distances = self.get_crowding_distances_at_fronts(fronts)
            new_positions = update_strategy_1.update_population(ranks, fronts, crowding_distances, self.individuals)
            # Update positions in population
            for k in range(self.n_individuals):
                self.individuals[k].x = new_positions[k]

        elif update_strategy == "strategy_2":
            update_strategy_2.update_population(ranks, fronts, self.objective_values, self.individuals)

        elif update_strategy == "strategy_3":
            update_strategy_3.update_population(ranks, fronts, self.objective_values, self.individuals)


        if boundary_strategy == "strategy_1":
            for k in range(self.n_individuals):
                current_positions = self.individuals[k].x

                idx_low = current_positions < self.problem.xl
                current_positions[idx_low] = self.problem.xl[idx_low]

                idx_up = current_positions > self.problem.xu
                current_positions[idx_up] = self.problem.xl[idx_up]

                self.individuals[k].x = current_positions

        elif boundary_strategy == "strategy_2":
            # Check for out of bounds
            for k in range(len(self.individuals)):
                current_individual = self.individuals[k]

                # Replace individual if out of bounds
                if np.any(current_individual.x < self.problem.xl) or \
                        np.any(current_individual.x > self.problem.xu):
                    current_individual.x = self.problem.xl + \
                                       self.rng.uniform(0, 1, self.problem.xl.shape) * (
                                                   self.problem.xu - self.problem.xl)

    def get_crowding_distances_at_fronts(self, fronts):
        crowding_distances_at_fronts = []
        n_fronts = len(fronts)
        for k in range(n_fronts):
            individuals_at_current_front = fronts[k]
            objectives_for_front = self.objective_values[individuals_at_current_front, :]
            crowding_distances_at_fronts.append(self.crowding_distance_calculation(objectives_for_front))
        return crowding_distances_at_fronts

    def check_termination(self):
        return False

    @staticmethod
    def nondominated_sorting(objective_values):
        n_individuals = objective_values.shape[0]
        domination_counts = np.zeros(n_individuals, dtype=int)
        ranks = -np.ones(n_individuals, dtype=int)
        dominated_individuals = [ [] for _ in range(n_individuals) ]


        for k in range(n_individuals):
            fk = objective_values[k, :]
            for l in range(n_individuals):
                if k == l:
                    continue

                fl = objective_values[l, :]
                # Solution k dominates solution p
                if np.all(fk <= fl) and np.any(fk < fl):
                    # Set of solutions dominated by solution k
                    dominated_individuals[k].append(l)

                    # Number of solutions that dominate solution l
                    domination_counts[l] += 1

        pareto_fronts = []
        current_pareto_front = []
        rank_counter = 0
        Q_next = []
        Q = np.where(domination_counts == 0)[0] # Find first front
        while len(Q) > 0:
            for k in Q:
                current_pareto_front.append(k)
                ranks[k] = rank_counter
                # Decrease domination counts
                for l in dominated_individuals[k]:
                    domination_counts[l] -= 1
                    # Find individuals whose domination counters just became 0
                    if domination_counts[l] == 0:
                        Q_next.append(l)

            Q = Q_next.copy()
            Q_next.clear()
            pareto_fronts.append(current_pareto_front.copy())
            current_pareto_front.clear()
            rank_counter = rank_counter + 1


        return pareto_fronts, ranks

    @staticmethod
    def crowding_distance_calculation(objective_values_of_pareto_front: np.ndarray):
        n_individuals_on_front = objective_values_of_pareto_front.shape[0]
        n_objectives = objective_values_of_pareto_front.shape[1]

        distances = np.zeros(n_individuals_on_front)
        for k in range(n_objectives):
            f_max = np.max(objective_values_of_pareto_front[:, k])
            f_min = np.min(objective_values_of_pareto_front[:, k])

            sorted_individuals = np.argsort(objective_values_of_pareto_front[:, k])
            sorted_objective = objective_values_of_pareto_front[sorted_individuals, k]

            distances_tmp = np.zeros(n_individuals_on_front)
            distances_tmp[0] = np.inf
            distances_tmp[-1] = np.inf

            for l in range(1, n_individuals_on_front - 1):
                distances_tmp[l] = (sorted_objective[l+1] - sorted_objective[l-1]) / (f_max - f_min)

            # Re-map to un-sorted individuals
            for m in range(n_individuals_on_front):
                distances[sorted_individuals[m]] = distances[sorted_individuals[m]] + distances_tmp[m]

        return distances

    def run(self, n_individuals: int,
            n_cycles: int = 100,
            rng_seed: int = 1234,
            callback: callable = None,
            update_strategy: str = "strategy_1",
            boundary_strategy: str = "strategy_1",
            *args, **kwargs):

        self.n_individuals = n_individuals
        self.rng = default_rng(rng_seed)
        self.create_initial_population(n_individuals)
        self.objective_values = np.zeros((n_individuals, self.n_obj))
        self.callback = callback

        terminate = False
        counter = 0

        while(not terminate and counter < n_cycles):
            print("Step {:3d}".format(counter))
            self.evaluate(*args, **kwargs)
            self.update_population(update_strategy, boundary_strategy)
            terminate = self.check_termination()
            counter = counter + 1

        return True
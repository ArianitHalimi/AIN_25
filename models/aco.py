import random
import copy
import math
from collections import defaultdict
from models.solution import Solution

class ACOOptimizer:
    """
    Ant Colony Optimization (ACO) for refining a given library signup solution.

    Accepts an initial solution (sequence of libraries) and tries to improve the order
    to maximize the scanned-book score under the given time constraint.
    """
    def __init__(
        self,
        data,
        initial_solution: Solution,
        num_ants: int = 50,
        num_iterations: int = 50,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.1,
        pheromone_deposit: float = 1.0
    ):
        self.data = data
        # Only optimize the libraries already in the solution
        self.library_order = (
            list(initial_solution.unsigned_libraries) +
            list(initial_solution.signed_libraries)
        )
        # Best solution so far
        self.best_solution = copy.deepcopy(initial_solution)
        self.best_score = initial_solution.fitness_score

        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit

        # Pheromone trails: tau[i][j] = pheromone on edge i->j
        self.tau = defaultdict(lambda: defaultdict(lambda: 1.0))
        # Static heuristic: eta[j] = quality of choosing library j next (start-independent)
        self.eta = self._compute_static_heuristic()

    def _compute_static_heuristic(self):
        """
        Compute a heuristic value for each library j:
        average book score / signup_days
        """
        eta = {}
        for lib in self.data.libs:
            if lib.id in self.library_order:
                scores = [self.data.scores[b.id] for b in lib.books]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                eta[lib.id] = avg_score / max(lib.signup_days, 1)
        return eta

    def optimize(self):
        """
        Run the ACO iterations and return the best refined Solution.
        """
        for iteration in range(self.num_iterations):
            all_solutions = []
            # Each ant builds a tour
            for _ in range(self.num_ants):
                tour = self._construct_tour()
                sol = self._rebuild_solution(tour)
                all_solutions.append((tour, sol))
                # Update global best
                if sol.fitness_score > self.best_score:
                    self.best_score = sol.fitness_score
                    self.best_solution = sol

            # Pheromone evaporation
            for i in self.tau:
                for j in self.tau[i]:
                    self.tau[i][j] *= (1 - self.evaporation_rate)

            # Deposit pheromones based on the best tour this iteration
            # find best among all_solutions
            best_tour, best_sol = max(all_solutions, key=lambda x: x[1].fitness_score)
            deposit_amount = self.pheromone_deposit / (1 + (best_sol.fitness_score))
            prev = 0  # start node id 0 index
            for j in best_tour:
                self.tau[prev][j] += deposit_amount
                prev = j

        return self.best_solution

    def _construct_tour(self):
        """
        Construct a single ant's tour (ordering of library ids) probabilistically
        based on pheromone and heuristic values.
        """
        unvisited = set(self.library_order)
        tour = []
        current = None  # start node
        while unvisited:
            # compute probabilities for next library
            weights = []
            for j in unvisited:
                pher = self.tau[current][j]
                heur = self.eta[j]
                weights.append((j, (pher ** self.alpha) * (heur ** self.beta)))
            total = sum(w for _, w in weights)
            if total <= 0:
                # fallback: random choice
                j = random.choice(list(unvisited))
            else:
                # roulette wheel selection
                r = random.random() * total
                cumulative = 0.0
                for j, w in weights:
                    cumulative += w
                    if cumulative >= r:
                        break
            tour.append(j)
            unvisited.remove(j)
            current = j
        return tour

    def _rebuild_solution(self, order: list[int]) -> Solution:
        """
        Given a signup order of library ids, rebuild the full Solution by greedily
        picking the highest-score unscanned books for each library in sequence.
        """
        curr_time = 0
        scanned = set()
        scanned_per_lib = {}
        signed = []
        for lib_id in order:
            lib = next(l for l in self.data.libs if l.id == lib_id)
            if curr_time + lib.signup_days >= self.data.num_days:
                break
            curr_time += lib.signup_days
            time_left = self.data.num_days - curr_time
            capacity = time_left * lib.books_per_day
            # pick top remaining by score
            available = sorted(
                (b.id for b in lib.books if b.id not in scanned),
                key=lambda b: -self.data.scores[b]
            )[:capacity]
            if not available:
                continue
            signed.append(lib_id)
            scanned_per_lib[lib_id] = available
            scanned.update(available)
        sol = Solution(signed, [], scanned_per_lib, scanned)
        sol.calculate_fitness_score(self.data.scores)
        return sol

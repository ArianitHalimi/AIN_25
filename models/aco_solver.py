import random
import copy
import math
from .solver import Solver
from .solution import Solution

class AcoSolver:
    def __init__(self, base_solver=None):
        """
        base_solver: an instance of your old Solver class 
                     (or pass None and create it inside)
        """
        if base_solver is None:
            self.base_solver = Solver()
        else:
            self.base_solver = base_solver

        # We'll store our pheromones in this list (one value per library)
        self.pheromones = []

    # -------------- Pheromone Logic (Algorithm 111) -------------
    def init_pheromones(self, data, initial_value=0.1):
        """Create pheromone array, one per library."""
        self.pheromones = [initial_value] * data.num_libs

    def pheromone_update(self, population, alpha=0.1):
        """
        Implementation of Algorithm 111:
          p[i] <- (1 - alpha) * p[i] + alpha * (r[i]/c[i]),
        where r[i] is sum of fitness of solutions containing library i,
        and c[i] is how many solutions used library i.
        """
        n = len(self.pheromones)
        r = [0.0] * n
        c = [0]   * n

        # accumulate
        for sol in population:
            fit = sol.fitness_score
            for lib_id in sol.signed_libraries:
                r[lib_id] += fit
                c[lib_id] += 1

        # update
        for i in range(n):
            if c[i] > 0:
                avg_desir = r[i] / c[i]
                self.pheromones[i] = (1 - alpha)*self.pheromones[i] + alpha*avg_desir

    # -------------- Generate solutions using pheromones -------------
    def generate_solution_aco(self, data):
        """Build a solution with probability influenced by pheromones."""
        if not self.pheromones:
            self.init_pheromones(data)

        lib_ids = list(range(data.num_libs))
        random.shuffle(lib_ids)

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for lib_id in lib_ids:
            lib = data.libs[lib_id]
            # pick prob
            prob = self.pheromones[lib_id] / (1 + self.pheromones[lib_id])
            if random.random() < prob:
                # attempt to sign library
                if curr_time + lib.signup_days < data.num_days:
                    signed_libraries.append(lib_id)
                    curr_time += lib.signup_days
                    time_left = data.num_days - curr_time
                    max_scan = time_left * lib.books_per_day
                    available_books = sorted(
                        {b.id for b in lib.books} - scanned_books,
                        key=lambda b: -data.scores[b]
                    )[:max_scan]
                    if available_books:
                        scanned_books_per_library[lib_id] = available_books
                        scanned_books.update(available_books)
                else:
                    unsigned_libraries.append(lib_id)
            else:
                unsigned_libraries.append(lib_id)

        sol = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)
        sol.calculate_fitness_score(data.scores)
        return sol

    def combined_tweak_seven_tasks(self, solution, data):
        """
        Combine your old 4 tasks from the base_solver + 
        these 3 new tasks => total 7. 
        We'll randomly pick one each time.
        """
        # let's gather references to the old tasks from your base_solver
        possible_tasks = [
            self.base_solver.tweak_solution_swap_signed,    
            self.base_solver.tweak_solution_swap_signed_with_unsigned, 
            self.base_solver.tweak_solution_swap_same_books, 
            self.base_solver.tweak_solution_swap_last_book,  
            self.base_solver.crossover
        ]
        chosen_task = random.choice(possible_tasks)
        return chosen_task(copy.deepcopy(solution), data)

    # -------------- The main ACO loop using 7 tasks + ILS initializer -------------
    def run_algorithm111(self, data, alpha=0.1, iterations=1000, population_size=5):
        """
        1. Initialize pheromones
        2. Use old solver's ILS as an initializer
        3. For each iteration, build population using:
            * some solutions from generate_solution_aco
            * some solutions from combined_tweak_seven_tasks
           Then update pheromones and track best.
        """
        # 1) init
        self.init_pheromones(data)

        # 2) get a base solution 
        base_score, base_sol = self.base_solver.iterated_local_search(data, time_limit=10)
        best_sol = base_sol

        for it in range(iterations):
            # build population
            population = []
            for _ in range(population_size):
                if random.random() < 0.5:
                    # from scratch using pheromones
                    sol = self.generate_solution_aco(data)
                    population.append(sol)
                else:
                    # from best by a 7-task tweak
                    neighbor = self.combined_tweak_seven_tasks(copy.deepcopy(best_sol), data)
                    population.append(neighbor)

            # update pheromones
            self.pheromone_update(population, alpha=alpha)

            # track best
            local_best = max(population, key=lambda s: s.fitness_score)
            if local_best.fitness_score > best_sol.fitness_score:
                best_sol = local_best
            #print(f"Iter {it+1}, best so far = {best_sol.fitness_score}")

        return best_sol

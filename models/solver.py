import random
from collections import defaultdict
import time
from models.library import Library
import os
# from tqdm import tqdm
from models.solution import Solution
import copy
import random
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from typing import Tuple
from models.instance_data import InstanceData
# Simulated Annealing Hybrid Paraelelism Cooling Functions

def cooling_exponential(temp, cooling_rate=0.003):
    return temp * (1 - cooling_rate)
def cooling_geometric(temp, alpha=0.95):
    return temp * alpha
def cooling_lundy_mees(temp, beta=0.001):
    return temp / (1 + beta * temp)

def _pool_init(instance_data: InstanceData, hc_steps: int, mutation_prob: float):
    global INSTANCE, HC_STEPS, MUT_PROB, SOLVER
    INSTANCE    = instance_data
    HC_STEPS    = hc_steps
    MUT_PROB    = mutation_prob
    SOLVER      = Solver()

def _process_offspring(sol: Solution) -> Solution:
    """In‐place mutation + hill‐climb on one offspring."""
    if random.random() < MUT_PROB:
        _, sol = SOLVER.hill_climbing_combined_w_initial_solution(sol, INSTANCE, iterations=HC_STEPS)
    return sol

class Solver:
    def __init__(self):
        pass
    
    def generate_initial_solution(self, data):
        Library._id_counter = 0
        
        shuffled_libs = data.libs.copy()
        random.shuffle(shuffled_libs)

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        # for library in tqdm(shuffled_libs): # If the visualisation is needed
        for library in shuffled_libs:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books, key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        solution = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)

        solution.calculate_fitness_score(data.scores)

        return solution

    def crossover(self, solution, data):
        """Performs crossover by shuffling library order and swapping books accordingly."""
        new_solution = copy.deepcopy(solution) 

        old_order = new_solution.signed_libraries[:]
        library_indices = list(range(len(data.libs)))
        random.shuffle(library_indices)

        new_scanned_books_per_library = {}

        for new_idx, new_lib_idx in enumerate(library_indices):
            if new_idx >= len(old_order):
                break 

            old_lib_id = old_order[new_idx]
            new_lib_id = new_lib_idx

            if new_lib_id < 0 or new_lib_id >= len(data.libs):
                print(f"Warning: new_lib_id {new_lib_id} is out of range for data.libs (size: {len(data.libs)})")
                continue

            if old_lib_id in new_solution.scanned_books_per_library:
                books_to_move = new_solution.scanned_books_per_library[old_lib_id]

                existing_books_in_new_lib = {book.id for book in data.libs[new_lib_id].books}

                valid_books = []
                for book_id in books_to_move:
                    if book_id not in existing_books_in_new_lib and book_id not in [b for b in valid_books]:
                        valid_books.append(book_id)

                new_scanned_books_per_library[new_lib_id] = valid_books

        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def hill_climbing_with_crossover(self, initial_solution, data):
        current_solution = initial_solution
        max_iterations = 100 
        convergence_threshold = 0.01
        last_fitness = current_solution.fitness_score

        start_time = time.time()

        for iteration in range(max_iterations):
            if iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, Fitness: {current_solution.fitness_score}")

            neighbor_solution = self.crossover(current_solution, data)

            if abs(neighbor_solution.fitness_score - last_fitness) < convergence_threshold:
                print(f"Converged with fitness score: {neighbor_solution.fitness_score}")
                break  
            current_solution = neighbor_solution
            last_fitness = neighbor_solution.fitness_score

        end_time = time.time()
        print(f"Total Time: {end_time - start_time:.2f} seconds")

        return current_solution

    def tweak_solution_swap_signed(self, solution, data):
        """
        Randomly swaps two libraries within the signed libraries list.
        This creates a new solution by exchanging the positions of two libraries
        while maintaining the feasibility of the solution.

        Args:
            solution: The current solution to tweak
            data: The problem data

        Returns:
            A new solution with two libraries swapped
        """
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)

        idx1, idx2 = random.sample(range(len(solution.signed_libraries)), 2)

        lib_id1 = solution.signed_libraries[idx1]
        lib_id2 = solution.signed_libraries[idx2]

        new_signed_libraries = solution.signed_libraries.copy()
        new_signed_libraries[idx1] = lib_id2
        new_signed_libraries[idx2] = lib_id1

        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        for lib_id in new_signed_libraries:
            library = data.libs[lib_id]

            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = []
            for book in library.books:
                if (
                    book.id not in scanned_books
                    and len(available_books) < max_books_scanned
                ):
                    available_books.append(book.id)

            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books

        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def hill_climbing_swap_signed(self, data, iterations = 1000):
        solution = self.generate_initial_solution_grasp(data)
        for i in range(iterations):
            solution_clone = copy.deepcopy(solution)
            new_solution = self.tweak_solution_swap_signed(solution_clone, data)
            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return (solution.fitness_score, solution)

    # region Hill Climbing Signed & Unsigned libs
    def _extract_lib_id(self, libraries, library_index):
        return int(libraries[library_index][len("Library "):])

    def tweak_solution_swap_signed_with_unsigned(self, solution, data, bias_type=None, bias_ratio=2/3):
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        local_signed_libs = solution.signed_libraries.copy()
        local_unsigned_libs = solution.unsigned_libraries.copy()

        total_signed = len(local_signed_libs)

        # Bias
        if bias_type == "favor_first_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(0, total_signed // 2 - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        elif bias_type == "favor_second_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(total_signed // 2, total_signed - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        else:
            signed_idx = random.randint(0, total_signed - 1)

        unsigned_idx = random.randint(0, len(local_unsigned_libs) - 1)

        # signed_lib_id = self._extract_lib_id(local_signed_libs, signed_idx)
        # unsigned_lib_id = self._extract_lib_id(local_unsigned_libs, unsigned_idx)
        signed_lib_id = local_signed_libs[signed_idx]
        unsigned_lib_id = local_unsigned_libs[unsigned_idx]

        # Swap the libraries
        local_signed_libs[signed_idx] = unsigned_lib_id
        local_unsigned_libs[unsigned_idx] = signed_lib_id
        # print(f"swapped_signed_lib={unsigned_lib_id}")
        # print(f"swapped_unsigned_lib={unsigned_lib_id}")

        # Preserve the part before `signed_idx`
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        lib_lookup = {lib.id: lib for lib in data.libs}

        # Process libraries before the swapped index
        for i in range(signed_idx):
            # lib_id = self._extract_lib_id(solution.signed_libraries, i)
            lib_id = solution.signed_libraries[i]
            library = lib_lookup.get(lib_id)

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)

        # Recalculate from `signed_idx` onward
        new_signed_libraries = local_signed_libs[:signed_idx]

        for i in range(signed_idx, len(local_signed_libs)):
            # lib_id = self._extract_lib_id(local_signed_libs, i)
            lib_id = local_signed_libs[i]
            library = lib_lookup.get(lib_id)

            if curr_time + library.signup_days >= data.num_days:
                solution.unsigned_libraries.append(library.id)
                continue

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_signed_libraries.append(library.id)  # Not f"Library {library.id}"
                new_scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)

        # Update solution
        new_solution = Solution(new_signed_libraries, local_unsigned_libs, new_scanned_books_per_library, scanned_books)
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def hill_climbing_swap_signed_with_unsigned(self, data, iterations=1000):
        solution = self.generate_initial_solution_grasp(data)

        for i in range(iterations - 1):
            new_solution = self.tweak_solution_swap_signed_with_unsigned(solution, data)
            # new_solution = self.tweak_solution_signed_unsigned(solution, data, bias_type="favor_second_half")
            # new_solution = self.tweak_solution_signed_unsigned(solution, data, bias_type="favor_first_half", bias_ratio=3/4)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return (solution.fitness_score, solution)

    def random_search(self, data, iterations = 1000):
        solution = self.generate_initial_solution_grasp(data)

        for i in range(iterations - 1):
            new_solution = self.generate_initial_solution_grasp(data)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return (solution.fitness_score, solution)

    def tweak_solution_swap_same_books(self, solution, data):
        library_ids = [lib for lib in solution.signed_libraries if lib < len(data.libs)]

        if len(library_ids) < 2:
            return solution

        idx1 = random.randint(0, len(library_ids) - 1)
        idx2 = random.randint(0, len(library_ids) - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, len(library_ids) - 1)

        library_ids[idx1], library_ids[idx2] = library_ids[idx2], library_ids[idx1]

        ordered_libs = [data.libs[lib_id] for lib_id in library_ids]

        all_lib_ids = set(range(len(data.libs)))
        remaining_lib_ids = all_lib_ids - set(library_ids)
        for lib_id in sorted(remaining_lib_ids):
            ordered_libs.append(data.libs[lib_id])

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for library in ordered_libs:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b],
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def hill_climbing_swap_same_books(self, data, iterations = 1000):
        Library._id_counter = 0
        solution = self.generate_initial_solution_grasp(data)

        for i in range(iterations):
            new_solution = self.tweak_solution_swap_same_books(solution, data)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return (solution.fitness_score, solution)

    def hill_climbing_combined(self, data, iterations = 1000):
        solution = self.generate_initial_solution_grasp(data)

        list_of_climbs = [
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_same_books,
            self.tweak_solution_swap_signed,
            self.tweak_solution_swap_last_book,
            self.tweak_solution_swap_neighbor_libraries,
            self.tweak_solution_insert_library,
        ]

        for i in range(iterations - 1):
            # if i % 100 == 0:
            #     print('i',i)
            target_climb = random.choice(list_of_climbs)
            solution_copy = copy.deepcopy(solution)
            new_solution = target_climb(solution_copy, data) 

            if (new_solution.fitness_score > solution.fitness_score):
                solution = new_solution

        return (solution.fitness_score, solution)

    def tweak_solution_swap_last_book(self, solution, data):
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution  # No scanned or unsigned libraries, return unchanged solution

        # Pick a random library that has scanned books
        chosen_lib_id = random.choice(list(solution.scanned_books_per_library.keys()))
        scanned_books = solution.scanned_books_per_library[chosen_lib_id]

        if not scanned_books:
            return solution  # Safety check, shouldn't happen

        # Get the last scanned book from this library
        last_scanned_book = scanned_books[-1]  # Last book in the list

        # library_dict = {f"Library {lib.id}": lib for lib in data.libs}
        library_dict = {lib.id: lib for lib in data.libs}

        best_book = None
        best_score = -1

        for unsigned_lib in solution.unsigned_libraries:
            library = library_dict[unsigned_lib]  # O(1) dictionary lookup

            # Find the first unscanned book from this library
            for book in library.books:
                if book.id not in solution.scanned_books:  # O(1) lookup in set
                    if data.scores[book.id] > best_score:  # Only store the best
                        best_book = book.id
                        best_score = data.scores[book.id]
                    break  # Stop after the first valid book

        # Assign the best book found (or None if none exist)
        first_unscanned_book = best_book

        if first_unscanned_book is None:
            return solution  # No available unscanned books

        # Create new scanned books mapping (deep copy)
        new_scanned_books_per_library = {
            lib_id: books.copy() for lib_id, books in solution.scanned_books_per_library.items()
        }

        # Swap the books
        new_scanned_books_per_library[chosen_lib_id].remove(last_scanned_book)
        new_scanned_books_per_library[chosen_lib_id].append(first_unscanned_book)

        # Update the overall scanned books set
        new_scanned_books = solution.scanned_books.copy()
        new_scanned_books.remove(last_scanned_book)
        new_scanned_books.add(first_unscanned_book)

        # Create the new solution
        new_solution = Solution(
            signed_libs=solution.signed_libraries.copy(),
            unsigned_libs=solution.unsigned_libraries.copy(),
            scanned_books_per_library=new_scanned_books_per_library,
            scanned_books=new_scanned_books
        )

        # Recalculate fitness score
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def hill_climbing_swap_last_book(self, data, iterations=1000):
        solution = self.generate_initial_solution_grasp(data)

        for i in range(iterations - 1):
            new_solution = self.tweak_solution_swap_last_book(solution, data)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return (solution.fitness_score, solution)


    def iterated_local_search(self, data, time_limit=300, max_iterations=1000):
        """
        Implements Iterated Local Search (ILS) with Random Restarts
        Args:
            data: The problem data
            time_limit: Maximum time in seconds (default: 300s = 5 minutes)
            max_iterations: Maximum number of iterations (default: 1000)
        """
        min_time = 5
        max_time = min(60, time_limit)
        T = list(range(min_time, max_time + 1, 5))

        S = self.generate_initial_solution_grasp(data, p=0.05, max_time=20)
        
        print(f"Initial solution fitness: {S.fitness_score}")

        H = copy.deepcopy(S)
        Best = copy.deepcopy(S)
        
        # Create a pool of solutions to choose from as homebase
        solution_pool = [copy.deepcopy(S)]
        pool_size = 5  # Maximum number of solutions to keep in the pool

        start_time = time.time()
        total_iterations = 0

        while (
            total_iterations < max_iterations
            and (time.time() - start_time) < time_limit
        ):
            local_time_limit = random.choice(T)
            local_start_time = time.time()

            while (time.time() - local_start_time) < local_time_limit and (
                time.time() - start_time
            ) < time_limit:

                selected_tweak = self.choose_tweak_method()
                R = selected_tweak(copy.deepcopy(S), data)

                if R.fitness_score > S.fitness_score:
                    S = copy.deepcopy(R)

                if S.fitness_score >= data.calculate_upper_bound():
                    return (S.fitness_score, S)

                total_iterations += 1
                if total_iterations >= max_iterations:
                    break

            if S.fitness_score > Best.fitness_score:
                Best = copy.deepcopy(S)

            # Update the solution pool
            if S.fitness_score >= H.fitness_score:
                H = copy.deepcopy(S)
                # Add the improved solution to the pool
                solution_pool.append(copy.deepcopy(S))
                # Keep only the best solutions in the pool
                solution_pool.sort(key=lambda x: x.fitness_score, reverse=True)
                if len(solution_pool) > pool_size:
                    solution_pool = solution_pool[:pool_size]
            else:
                # Instead of random acceptance, choose a random solution from the pool
                if len(solution_pool) > 1:  # Only if we have more than one solution in the pool
                    H = copy.deepcopy(random.choice(solution_pool))
                # Add the current solution to the pool if it's not already there
                if S not in solution_pool:
                    solution_pool.append(copy.deepcopy(S))
                    # Keep only the best solutions in the pool
                    solution_pool.sort(key=lambda x: x.fitness_score, reverse=True)
                    if len(solution_pool) > pool_size:
                        solution_pool = solution_pool[:pool_size]

            S = self.perturb_solution(H, data)

            if Best.fitness_score >= data.calculate_upper_bound():
                break

        return (Best.fitness_score, Best)

    def perturb_solution(self, solution, data):
        """Helper method for ILS to perturb solutions with destroy-and-rebuild strategy"""
        perturbed = copy.deepcopy(solution)

        max_destroy_size = len(perturbed.signed_libraries)
        if max_destroy_size == 0:
            return perturbed

        destroy_size = random.randint(
            min(1, max_destroy_size), min(max_destroy_size, max_destroy_size // 3 + 1)
        )

        libraries_to_remove = random.sample(perturbed.signed_libraries, destroy_size)

        new_signed_libraries = [
            lib for lib in perturbed.signed_libraries if lib not in libraries_to_remove
        ]
        new_unsigned_libraries = perturbed.unsigned_libraries + libraries_to_remove

        new_scanned_books = set()
        new_scanned_books_per_library = {}

        for lib_id in new_signed_libraries:
            if lib_id in perturbed.scanned_books_per_library:
                new_scanned_books_per_library[lib_id] = (
                    perturbed.scanned_books_per_library[lib_id].copy()
                )
                new_scanned_books.update(new_scanned_books_per_library[lib_id])

        curr_time = sum(
            data.libs[lib_id].signup_days for lib_id in new_signed_libraries
        )

        lib_scores = []
        for lib_id in new_unsigned_libraries:
            library = data.libs[lib_id]
            available_books = [
                b for b in library.books if b.id not in new_scanned_books
            ]
            if not available_books:
                continue
            avg_score = sum(data.scores[b.id] for b in available_books) / len(
                available_books
            )
            score = library.books_per_day * avg_score / library.signup_days
            lib_scores.append((score, lib_id))

        lib_scores.sort(reverse=True)

        for _, lib_id in lib_scores:
            library = data.libs[lib_id]

            if curr_time + library.signup_days >= data.num_days:
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b],
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
                new_unsigned_libraries.remove(lib_id)

        rebuilt_solution = Solution(
            new_signed_libraries,
            new_unsigned_libraries,
            new_scanned_books_per_library,
            new_scanned_books,
        )
        rebuilt_solution.calculate_fitness_score(data.scores)

        return rebuilt_solution

    def max_possible_score(self, data):

        return sum(book.score for book in data.books.values())

    def hill_climbing_with_random_restarts(self, data, total_time_ms=1000):
        Library._id_counter = 0
    # Lightweight solution representation
        def create_light_solution(solution):
            return {
                "signed": list(solution.signed_libraries),
                "books": dict(solution.scanned_books_per_library),
                "score": solution.fitness_score
            }

        # Initialize
        current = create_light_solution(self.generate_initial_solution_grasp(data))
        best = current.copy()
        tweak_functions = [
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_signed,
            self.tweak_solution_swap_last_book
        ]
        
        # Adaptive parameters
        tweak_weights = [1.0] * 3  # Initial weights for 3 tweaks
        tweak_success = [0] * 3
        temperature = 1000  # Controls solution acceptance
        stagnation = 0      # Iterations since last improvement
        
        # Time management
        start_time = time.time()
        time_distribution = [100, 200, 300, 400, 500]  # ms - possible time intervals for each restart

        while (time.time() - start_time) * 1000 < total_time_ms:
            # Set time limit for this restart
            time_limit = (time.time() + random.choice(time_distribution) / 1000)
            
            # Reset current solution for this restart
            current = create_light_solution(self.generate_initial_solution_grasp(data))
            temperature = 1000  # Reset temperature for each restart
            
            # Inner loop for this restart period
            while (time.time() - start_time) * 1000 < total_time_ms and time.time() < time_limit:
                # 1. Select tweak function dynamically
                total_weight = sum(tweak_weights)
                r = random.uniform(0, total_weight)
                tweak_idx = 0
                while r > tweak_weights[tweak_idx]:
                    r -= tweak_weights[tweak_idx]
                    tweak_idx += 1

                # 2. Generate neighbor (avoid deepcopy)
                neighbor = create_light_solution(
                    tweak_functions[tweak_idx](
                        Solution(current["signed"], [], current["books"], set()),
                        data
                    )
                )

                # 3. Simulated annealing acceptance
                delta = neighbor["score"] - current["score"]
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current = neighbor
                    tweak_success[tweak_idx] += 1

                    # Update best solution
                    if current["score"] > best["score"]:
                        best = current.copy()
                        stagnation = 0
                    else:
                        stagnation += 1

                # 4. Adaptive tweak weights update
                if random.random() < 0.01:  # Small chance to update weights
                    for i in range(3):
                        success_rate = tweak_success[i] / (sum(tweak_success) + 1)
                        tweak_weights[i] = max(0.5, min(5.0, tweak_weights[i] * (0.9 + success_rate)))
                    tweak_success = [0] * 3

                # 5. Cool temperature to reduce exploration over time
                temperature *= 0.995

        # Convert back to full solution
        return best["score"], Solution(
            best["signed"],
            [],
            best["books"],
            {b for books in best["books"].values() for b in books}
        )

    def _get_signature(self, solution):
        return tuple(solution.signed_libraries)
    
    def tabu_search(self, initial_solution, data, tabu_max_len=10, n=5, max_iterations=100):
        S = copy.deepcopy(initial_solution)
        S.calculate_fitness_score(data.scores)
        Best = copy.deepcopy(S)
        
        L = deque(maxlen=tabu_max_len)
        L.append(self._get_signature(S))

        for iteration in range(max_iterations):
            print(f"Iteration {iteration+1}, Current: {S.fitness_score}, Best: {Best.fitness_score}")

            R = self.tweak_solution_swap_last_book(S, data)

            for _ in range(n - 1):
                W = self.tweak_solution_swap_last_book(S, data)
                sig_W = self._get_signature(W)
                sig_R = self._get_signature(R)

                if (sig_W not in L and W.fitness_score > R.fitness_score) or (sig_R in L):
                    R = W 

            sig_R = self._get_signature(R)

            if sig_R not in L and R.fitness_score > S.fitness_score:
                S = R 

            L.append(sig_R)

            if S.fitness_score > Best.fitness_score:
                Best = copy.deepcopy(S)

        return Best
    
    def feature_based_tabu_search(self, initial_solution, data, tabu_max_len=10, n=5, max_iterations=100):
        from time import perf_counter

        S = copy.deepcopy(initial_solution)
        S.calculate_fitness_score(data.scores)
        Best = copy.deepcopy(S)

        L = deque(maxlen=tabu_max_len)  # Stores (signature, timestamp)
        tabu_set = set()  # Faster lookup than looping over deque
        c = 0
        sig_S = self._get_signature(S)
        L.append((sig_S, c))
        tabu_set.add(sig_S)

        tweak_functions = [
            self.tweak_solution_swap_last_book,
            self.tweak_solution_swap_signed,
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_same_books
        ]

        def tweak_avoiding_tabu(S_ref, L_set):
            max_attempts = 10
            for _ in range(max_attempts):
                tweak = random.choice(tweak_functions)
                try:
                    S_copy = copy.deepcopy(S_ref)
                    R = tweak(S_copy, data)
                    sig = self._get_signature(R)
                    if sig not in L_set:
                        return R, [sig]
                except:
                    continue
            # fallback
            fallback_copy = copy.deepcopy(S_ref)
            return fallback_copy, [self._get_signature(fallback_copy)]

        for iteration in range(max_iterations):
            c += 1
            # Clean old tabu entries
            L = deque([(feature, ts) for feature, ts in L if c - ts <= tabu_max_len], maxlen=tabu_max_len)
            tabu_set = set(f for f, _ in L)

            R, modified_features_R = tweak_avoiding_tabu(S, tabu_set)

            for _ in range(n - 1):
                W, modified_features_W = tweak_avoiding_tabu(S, tabu_set)
                if W.fitness_score > R.fitness_score:
                    R = W
                    modified_features_R = modified_features_W

            if R.fitness_score > S.fitness_score:
                S = R
                for feature in modified_features_R:
                    L.append((feature, c))
                    tabu_set.add(feature)

                if S.fitness_score > Best.fitness_score:
                    Best = copy.deepcopy(S)

            # Optional: remove this if you don't need runtime feedback
            # print(f"Iter {iteration}: Current = {S.fitness_score}, Best = {Best.fitness_score}")

        return Best

    def simulated_annealing_with_cutoff(self, data, total_time_ms=1000, max_steps=10000):
        # Lightweight solution representation
        def create_light_solution(solution):
            return {
                "signed": list(solution.signed_libraries),
                "books": dict(solution.scanned_books_per_library),
                "score": solution.fitness_score
            }

        # Initialize
        current = create_light_solution(self.generate_initial_solution_grasp(data))
        best = current.copy()
        tweak_functions = [
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_signed,
            self.tweak_solution_swap_last_book
        ]

        # Adaptive parameters
        tweak_weights = [1.0] * 3  # Initial weights for 3 tweaks
        tweak_success = [0] * 3
        temperature = 1000  # Controls solution acceptance
        stagnation = 0  # Iterations since last improvement

        # Time management
        start_time = time.time()

        steps_taken = 0  # To track the number of steps taken
        while (time.time() - start_time) * 1000 < total_time_ms and steps_taken < max_steps:
            # 1. Select tweak function dynamically
            total_weight = sum(tweak_weights)
            r = random.uniform(0, total_weight)
            tweak_idx = 0
            while r > tweak_weights[tweak_idx]:
                r -= tweak_weights[tweak_idx]
                tweak_idx += 1

            # 2. Generate neighbor (avoid deepcopy)
            neighbor = create_light_solution(
                tweak_functions[tweak_idx](
                    Solution(current["signed"], [], current["books"], set()),
                    data
                )
            )

            # 3. Simulated annealing acceptance
            delta = neighbor["score"] - current["score"]
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current = neighbor
                tweak_success[tweak_idx] += 1

                # Update best solution
                if current["score"] > best["score"]:
                    best = current.copy()
                    stagnation = 0
                else:
                    stagnation += 1

            # 4. Adaptive tweak weights update
            if random.random() < 0.01:  # Small chance to update weights
                for i in range(3):
                    success_rate = tweak_success[i] / (sum(tweak_success) + 1)
                    tweak_weights[i] = max(0.5, min(5.0, tweak_weights[i] * (0.9 + success_rate)))
                tweak_success = [0] * 3

            # 5. Cool temperature to reduce exploration over time
            temperature *= 0.995

            # Increment step counter
            steps_taken += 1

        # Convert back to full solution
        return best["score"], Solution(
            best["signed"],
            [],
            best["books"],
            {b for books in best["books"].values() for b in books}
        )
    
    def monte_carlo_search(self, data, num_iterations=1000, time_limit=None):
        """
        Monte Carlo search algorithm for finding optimal library configurations.
        
        Args:
            data: The problem instance data
            num_iterations: Maximum number of iterations to perform
            time_limit: Maximum time to run in seconds (optional)
            
        Returns:
            Tuple of (best_score, best_solution)
        """
        best_solution = None
        best_score = 0
        start_time = time.time()
        
        for i in range(num_iterations):
            # Check time limit if specified
            if time_limit and time.time() - start_time > time_limit:
                break
                
            # Generate a random solution
            current_solution = self.generate_initial_solution_grasp(data)
            
            # Evaluate the solution
            current_score = current_solution.fitness_score
            
            # Update best solution if current is better
            if current_score > best_score:
                best_score = current_score
                best_solution = current_solution
                
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Best Score: {best_score:,}")
                
        return best_score, best_solution

    def steepest_ascent_hill_climbing(self, data, total_time_ms=1000, n=5):
        start_time = time.time() * 1000
        current_solution = self.generate_initial_solution_grasp(data)
        best_solution = current_solution
        best_score = current_solution.fitness_score
        
        while (time.time() * 1000 - start_time) < total_time_ms:
            R = self.tweak_solution_swap_signed(copy.deepcopy(current_solution), data)
            best_tweak = R
            best_tweak_score = R.fitness_score
            
            for _ in range(n - 1):
                if (time.time() * 1000 - start_time) >= total_time_ms:
                    break
                
                W = self.tweak_solution_swap_signed(copy.deepcopy(current_solution), data)
                current_score = W.fitness_score
                if current_score > best_tweak_score:
                    best_tweak = W
                    best_tweak_score = current_score
            
            
            if best_tweak_score > best_score:
                current_solution = copy.deepcopy(best_tweak)
                best_score = best_tweak_score
                best_solution = current_solution
        
        return best_score, best_solution
    
    def best_of_steepest_ascent_and_random_restart(self, data, total_time_ms=1000):
        start_time = time.time() * 1000  # Start time in milliseconds
        time_steepest = total_time_ms // 2
        steepest_score, steepest_sol = self.steepest_ascent_hill_climbing(data, total_time_ms=time_steepest, n=5)

        elapsed_time = time.time() * 1000 - start_time
        remaining_time = max(0, total_time_ms - elapsed_time)

        restarts_score, restarts_sol = self.hill_climbing_with_random_restarts(data, total_time_ms=remaining_time)

        if steepest_score >= restarts_score:
            print("steepest ascent algorithm chosen: ", steepest_score)
            return steepest_score, steepest_sol
        else:
            print("random restart algorithm chosen: ", restarts_score)
            return restarts_score, restarts_sol
    

    
    def build_grasp_solution(self, data, p=0.05):
        """
        Build a feasible solution using a GRASP-like approach:
        - Sorting libraries by signup_days ASC, then total_score DESC.
        - Repeatedly choosing from the top p% feasible libraries at random.

        Args:
            data: The problem data (libraries, scores, num_days, etc.)
            p: Percentage (as a fraction) for the restricted candidate list (RCL)

        Returns:
            A Solution object with the constructed solution
        """
        libs_sorted = sorted(
            data.libs,
            key=lambda l: (l.signup_days, -sum(data.scores[b.id] for b in l.books)),
        )

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        candidate_libs = libs_sorted[:]

        while candidate_libs:
            rcl_size = max(1, int(len(candidate_libs) * p))
            rcl = candidate_libs[:rcl_size]

            chosen_lib = random.choice(rcl)
            candidate_libs.remove(chosen_lib)

            if curr_time + chosen_lib.signup_days >= data.num_days:
                unsigned_libraries.append(chosen_lib.id)
            else:
                time_left = data.num_days - (curr_time + chosen_lib.signup_days)
                max_books_scanned = time_left * chosen_lib.books_per_day

                available_books = sorted(
                    {book.id for book in chosen_lib.books} - scanned_books,
                    key=lambda b: -data.scores[b],
                )[:max_books_scanned]

                if available_books:
                    signed_libraries.append(chosen_lib.id)
                    scanned_books_per_library[chosen_lib.id] = available_books
                    scanned_books.update(available_books)
                    curr_time += chosen_lib.signup_days
                else:
                    unsigned_libraries.append(chosen_lib.id)

        solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        solution.calculate_fitness_score(data.scores)
        return solution

    def generate_initial_solution_grasp(self, data, p=0.05, max_time=60):
        """
        Generate an initial solution using a GRASP-like approach:
        1) Sort libraries by (signup_days ASC, total_score DESC).
        2) Repeatedly pick from top p% of feasible libraries at random.
        3) Optionally improve with a quick local search for up to max_time seconds.

        :param data:      The problem data (libraries, scores, num_days, etc.).
        :param p:         Percentage (as a fraction) for the restricted candidate list (RCL).
        :param max_time:  Time limit (in seconds) to repeat GRASP + local search.
        :return:          A Solution object with the best found solution.
        """
        start_time = time.time()
        best_solution = None
        Library._id_counter = 0

        while time.time() - start_time < max_time:
            candidate_solution = self.build_grasp_solution(data, p)

            improved_solution = self.local_search(
                candidate_solution, data, time_limit=1.0
            )

            if (best_solution is None) or (
                improved_solution.fitness_score > best_solution.fitness_score
            ):
                best_solution = improved_solution

        return best_solution

    def local_search(self, solution, data, time_limit=1.0):
        """
        A simple local search/hill-climbing method that randomly selects one of the available tweak methods.
        Uses choose_tweak_method to select the tweak operation based on defined probabilities.
        Runs for 'time_limit' seconds and tries small random modifications.
        """
        start_time = time.time()
        best = copy.deepcopy(solution)

        while time.time() - start_time < time_limit:
            selected_tweak = self.choose_tweak_method()

            neighbor = selected_tweak(copy.deepcopy(best), data)
            if neighbor.fitness_score > best.fitness_score:
                best = neighbor

        return best

    def choose_tweak_method(self):
        """Randomly chooses a tweak method based on the defined probabilities."""
        tweak_methods = [
            (self.tweak_solution_swap_signed_with_unsigned, 0.5),
            (self.tweak_solution_swap_same_books, 0.1),
            (self.crossover, 0.2),
            (self.tweak_solution_swap_last_book, 0.1),
            (self.tweak_solution_swap_signed, 0.1),
        ]

        methods, weights = zip(*tweak_methods)

        selected_method = random.choices(methods, weights=weights, k=1)[0]
        return selected_method

    def generate_initial_solution_sorted(self, data):
        """
        Generate an initial solution by sorting libraries by:
        1. Signup time in ascending order (fastest libraries first)
        2. Total book score in descending order (highest scoring libraries first)
        
        This deterministic approach prioritizes libraries that can be signed up quickly
        and have high total book scores.
        
        Args:
            data: The problem data containing libraries, books, and scores
            
        Returns:
            A Solution object with the constructed solution
        """
        Library._id_counter = 0
        # Sort libraries by signup time ASC and total book score DESC
        sorted_libraries = sorted(
            data.libs,
            key=lambda l: (l.signup_days, -sum(data.scores[b.id] for b in l.books))
        )
        
        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0
        
        for library in sorted_libraries:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                unsigned_libraries.append(library.id)
        
        solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books
        )
        solution.calculate_fitness_score(data.scores)
        
        return solution
    
    def variable_neighborhood_search(self, data, time_limit_ms=10000):
       
            start_time = time.time()
            time_limit_s = time_limit_ms / 1000.0

            current_solution = self.generate_initial_solution_grasp(data, p=0.05, max_time=5)
            best_score = current_solution.fitness_score

            operators = [
                self.tweak_solution_swap_signed_with_unsigned,
                self.tweak_solution_swap_signed,
                self.tweak_solution_swap_last_book,
                self.tweak_solution_swap_same_books
            ]

            k = 0
            while time.time() - start_time < time_limit_s:
                operator = operators[k]
                new_solution = operator(copy.deepcopy(current_solution), data)

                if new_solution.fitness_score > best_score:
                    current_solution = new_solution
                    best_score = new_solution.fitness_score
                    k = 0 
                else:
                    k += 1
                    if k >= len(operators):
                        k = 0 

            return best_score, current_solution


    def guided_local_search(self, data, max_time=300, max_iterations=1000):
        C = set(range(len(data.libs)))  # all possible library IDs
        
        T = list(range(5, 16, 5))  # Shorter time intervals (5, 10, 15 seconds) for more frequent updates
        
        p = [0] * len(data.libs)
        
        component_utilities = {
            i: sum(data.scores[book.id] for book in data.libs[i].books)
            for i in C
        }
        
        S = self.generate_initial_solution_grasp(data, p=0.1)  # more randomized initial solution
        
        Best = copy.deepcopy(S)
        
        start_time = time.time()
        stagnation_count = 0
        last_improvement_time = start_time
        iteration_count = 0
        
        # repeat until max time or iterations reached
        while time.time() - start_time < max_time and iteration_count < max_iterations:
            iteration_count += 1
            
            local_time_limit = time.time() + random.choice(T)
            local_best = copy.deepcopy(S)
            local_iterations = 0
            max_local_iterations = 50
            
            while time.time() < local_time_limit and local_iterations < max_local_iterations:
                local_iterations += 1
                
                for _ in range(3):
                    available_components = C - set(S.signed_libraries)
                    if available_components:
                        adjusted_utilities = {
                            c: component_utilities[c] / (1 + p[c])
                            for c in available_components
                        }
                        selected_component = max(adjusted_utilities.items(), key=lambda x: x[1])[0]
                        
                        tweak_function = random.choice([
                            self.tweak_solution_swap_signed_with_unsigned,
                            self.tweak_solution_swap_same_books,
                            self.tweak_solution_swap_signed,
                            self.tweak_solution_swap_last_book,
                            self.tweak_solution_swap_neighbor_libraries,
                            self.tweak_solution_insert_library
                        ])
                        R = tweak_function(copy.deepcopy(S), data)
                        
                        if tweak_function == self.tweak_solution_insert_library:
                            R = self.tweak_solution_insert_library(S, data, target_lib=selected_component)
                    else:
                        tweak_function = random.choice([
                            self.tweak_solution_swap_signed_with_unsigned,
                            self.tweak_solution_swap_same_books,
                            self.tweak_solution_swap_signed,
                            self.tweak_solution_swap_last_book,
                            self.tweak_solution_swap_neighbor_libraries
                        ])
                        R = tweak_function(copy.deepcopy(S), data)
                    
                    if R.fitness_score > Best.fitness_score:
                        Best = copy.deepcopy(R)
                        print(f"New best score: {Best.fitness_score:,} (iteration {iteration_count})")
                        last_improvement_time = time.time()
                        stagnation_count = 0
                    
                    R_quality = R.fitness_score - sum(p[i] for i in R.signed_libraries)
                    S_quality = S.fitness_score - sum(p[i] for i in S.signed_libraries)
                    if R_quality > S_quality:
                        S = copy.deepcopy(R)
                        if R.fitness_score > local_best.fitness_score:
                            local_best = copy.deepcopy(R)
                
                if (Best.fitness_score >= data.calculate_upper_bound() or 
                    time.time() - start_time >= max_time or 
                    iteration_count >= max_iterations):
                    break
            
            if local_best.fitness_score <= S.fitness_score:
                stagnation_count += 1
            
            C_prime = set()
            
            current_components = set(S.signed_libraries) & C
            for Ci in current_components:
                is_most_penalizable = True
                Ci_utility = component_utilities[Ci]
                Ci_penalizability = Ci_utility / (1 + p[Ci])
                
                for Cj in current_components:
                    if Cj != Ci:
                        Cj_utility = component_utilities[Cj]
                        Cj_penalizability = Cj_utility / (1 + p[Cj])
                        if Cj_penalizability > Ci_penalizability:
                            is_most_penalizable = False
                            break
                
                if is_most_penalizable:
                    C_prime.add(Ci)
            
            for Ci in C_prime:
                p[Ci] += 1
            
            if stagnation_count > 10:
                print(f"Resetting search due to stagnation (iteration {iteration_count})...")
                p = [0] * len(data.libs)
                S = self.generate_initial_solution_grasp(data, p=0.2)
                stagnation_count = 0
            
            if len(C_prime) > 0:
                elapsed = time.time() - start_time
                print(f"Time: {elapsed:.2f}s, Iteration: {iteration_count}/{max_iterations}, Best score: {Best.fitness_score:,}")
                print(f"Penalized {len(C_prime)} components, Total penalties: {sum(p)}")
            
            if (Best.fitness_score >= data.calculate_upper_bound() or 
                time.time() - last_improvement_time > 60):
                break
        
        print(f"Search completed after {time.time() - start_time:.2f} seconds and {iteration_count} iterations")
        return Best

    def hill_climbing_insert_library(self, data, iterations=1000):
        Library._id_counter = 0

        valid_library_ids = set(range(len(data.libs)))

        if isinstance(data.book_libs, list):
            book_libs_dict = {}
            for book_id, lib_ids in enumerate(data.book_libs):
                if isinstance(lib_ids, (list, tuple)):
                    book_libs_dict[book_id] = [
                        lib_id for lib_id in lib_ids
                        if lib_id in valid_library_ids
                    ]
            data.book_libs = book_libs_dict
        elif hasattr(data, 'book_libs') and isinstance(data.book_libs, dict):
            cleaned_book_libs = {}
            for book_id, lib_ids in data.book_libs.items():
                if isinstance(lib_ids, (list, tuple)):
                    cleaned_book_libs[book_id] = [
                        lib_id for lib_id in lib_ids
                        if lib_id in valid_library_ids
                    ]
            data.book_libs = cleaned_book_libs
        else:
            raise ValueError("data.book_libs must be either a list or dictionary")

        solution = self.generate_initial_solution(data)
        solution.unsigned_libraries = [
            lib_id for lib_id in solution.unsigned_libraries
            if lib_id in valid_library_ids
        ]

        for _ in range(iterations):
            new_solution = self.tweak_solution_insert_library(solution, data)
            
            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return solution.fitness_score, solution

    def tweak_solution_insert_library(self, solution, data, target_lib=None):
        if not solution.unsigned_libraries and target_lib is None:
            return solution

        new_solution = copy.deepcopy(solution)
        curr_time = sum(data.libs[lib_id].signup_days for lib_id in new_solution.signed_libraries)
        
        if target_lib is not None and target_lib not in new_solution.signed_libraries:
            lib_to_insert = target_lib
        else:
            if not new_solution.unsigned_libraries:
                return solution
            insert_idx = random.randint(0, len(new_solution.unsigned_libraries) - 1)
            lib_to_insert = new_solution.unsigned_libraries[insert_idx]
            new_solution.unsigned_libraries.pop(insert_idx)

        if curr_time + data.libs[lib_to_insert].signup_days >= data.num_days:
            return solution
            
        time_left = data.num_days - (curr_time + data.libs[lib_to_insert].signup_days)
        max_books_scanned = time_left * data.libs[lib_to_insert].books_per_day
        
        available_books = sorted(
            {book.id for book in data.libs[lib_to_insert].books} - new_solution.scanned_books,
            key=lambda b: -data.scores[b]
        )[:max_books_scanned]
        
        if available_books:
            best_pos = len(new_solution.signed_libraries)
            best_score = 0
            best_solution = None
            
            for pos in range(len(new_solution.signed_libraries) + 1):
                test_solution = copy.deepcopy(new_solution)
                test_solution.signed_libraries.insert(pos, lib_to_insert)
                test_solution.scanned_books_per_library[lib_to_insert] = available_books
                test_solution.scanned_books.update(available_books)
                test_solution.calculate_fitness_score(data.scores)
                
                if test_solution.fitness_score > best_score:
                    best_score = test_solution.fitness_score
                    best_pos = pos
                    best_solution = test_solution
            
            return best_solution if best_solution else solution
        
        return solution

    def tweak_solution_swap_neighbor_libraries(self, solution, data):
        """Swaps two adjacent libraries in the signed list to create a neighbor solution."""
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)
        swap_pos = random.randint(0, len(new_solution.signed_libraries) - 2)
        
        # Swap adjacent libraries
        new_solution.signed_libraries[swap_pos], new_solution.signed_libraries[swap_pos + 1] = \
            new_solution.signed_libraries[swap_pos + 1], new_solution.signed_libraries[swap_pos]
        
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        
        # Process libraries before swap point
        for i in range(swap_pos):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):  # Safety check
                continue
            library = data.libs[lib_id]
            curr_time += library.signup_days
            
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)
        
        # Re-process from swap point
        i = swap_pos
        while i < len(new_solution.signed_libraries):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):  # Skip invalid library IDs
                new_solution.unsigned_libraries.append(lib_id)
                new_solution.signed_libraries.pop(i)
                continue
                
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.extend(new_solution.signed_libraries[i:])
                new_solution.signed_libraries = new_solution.signed_libraries[:i]
                break
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
                i += 1
            else:
                new_solution.unsigned_libraries.append(lib_id)
                new_solution.signed_libraries.pop(i)
        
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)
        
        return new_solution
 
    def hill_climbing_swap_neighbors(self, data, iterations=1000):
        solution = self.generate_initial_solution(data)
        best_score = solution.fitness_score
         
        for _ in range(iterations):
            new_solution = self.tweak_solution_swap_neighbor_libraries(solution, data)
            
            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution
                best_score = solution.fitness_score
        
        return (best_score, solution)
    
    def hybrid_parallel_evolutionary_search(
        self,
        data: InstanceData,
        num_iterations: int = 1000,
        time_limit: float = None
    ) -> Tuple[float, Solution]:
        """
        Optimized hybrid GA: population-based crossover + parallel hill-climbing mutations,
        adaptive stagnation, and early stopping.
        """
        best_solution   = None
        best_score      = 0.0
        start_time      = time.time()
        stagnation_cnt  = 0
        max_stagnation  = 50
        
        population_size  = 4
        tour_size        = 2
        mutation_prob    = 0.3
        hill_climb_steps = 50

        # 1) Initialize population
        population = self.initialize_population(self.generate_initial_solution_grasp, data)

        # record initial best
        for sol in population:
            if sol.fitness_score > best_score:
                best_score, best_solution = sol.fitness_score, sol

        # 2) Launch pool once for all generations
        with ProcessPoolExecutor(
            max_workers=max(1, population_size // 2),
            initializer=_pool_init,
            initargs=(data, hill_climb_steps, mutation_prob)
        ) as executor:

            iteration = 0
            while iteration < num_iterations:
                # time limit?
                if time_limit and (time.time() - start_time) > time_limit:
                    break

                # sort & evaluate
                population.sort(key=lambda s: s.fitness_score, reverse=True)
                current_best = population[0]

                # update best / stagnation
                if current_best.fitness_score > best_score:
                    best_score, best_solution = current_best.fitness_score, current_best
                    stagnation_cnt = 0
                else:
                    stagnation_cnt += 1

                # early stop?
                if stagnation_cnt >= max_stagnation:
                    print(f"Early stopping at iteration {iteration} due to no improvement")
                    break

                # build next generation
                new_pop = [current_best]  # elitism

                # generate raw offspring
                raw_offspring = []
                while len(raw_offspring) < population_size - 1:
                    p1 = self.tournament_select(population)
                    p2 = self.tournament_select(population)
                    o1, o2 = self.crossover(p1, p2)
                    raw_offspring.append(o1)
                    if len(raw_offspring) < population_size - 1:
                        raw_offspring.append(o2)

                # parallel mutation + hill‑climb
                offspring = list(executor.map(_process_offspring, raw_offspring, chunksize=1))

                new_pop.extend(offspring)
                population = new_pop

                iteration += 1
                if iteration % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"Iteration {iteration}, Best Score: {best_score:,}, Time: {elapsed:.1f}s")

        # final fallback
        if best_solution is None:
            best_solution = self.generate_initial_solution_grasp(data)
            best_score    = best_solution.fitness_score

        return best_score, best_solution

    def initialize_population(self, initializer, data):
        """Initialize population using the provided initializer function."""
        population_size = 4
        return [initializer(data) for _ in range(population_size)]

    def tournament_select(self, population):
        """Select a solution using tournament selection."""
        tournament_size  = 2
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness_score)
    
    def great_deluge_algorithm(self, data, max_time=300, max_iterations=1000, delta_B=None):
        """
        Enhanced Great Deluge Algorithm with GRASP initialization and optimized parameters
        """
        # Validate input type
        if not hasattr(data, 'libs'):
            raise TypeError("First argument must be problem Data instance")

        # Initialize with GRASP-generated solution
        current_solution = self.generate_initial_solution_grasp(data, p=0.1, max_time=30)
        current_score = current_solution.fitness_score
        best_solution = copy.deepcopy(current_solution)
        best_score = current_score
        
        # Adaptive boundary initialization
        initial_boundary_buffer = 1.25  # Increased from 1.1 for better exploration
        B = current_score * initial_boundary_buffer
        
        # Dynamic decay calculation
        if delta_B is None:
            delta_B = (current_score * 0.3) / max_iterations  # More aggressive initial decay
            
        # Memory structures for stagnation detection
        memory_window = deque(maxlen=75)  # Larger window size for better trend detection
        improvement_threshold = current_score * 0.002  # More tolerant threshold
        
        # Nonlinear decay parameters
        alpha = 0.92  # Faster decay acceleration
        beta = 1.05   # Stronger exploration boost
        
        start_time = time.time()
        iterations = 0
        
        while (time.time() - start_time) < max_time and iterations < max_iterations:
            try:
                # Generate neighbor using combined hill climbing with adaptive depth
                neighbor = self.hill_climbing_combined(
                    data, 
                    iterations=int(15 * (1 - iterations/max_iterations))  # Increased initial depth
                )[1]
                
                # Solution validation
                if not isinstance(neighbor, Solution):
                    raise RuntimeError("Neighbor generation failed - invalid solution type")

                neighbor_score = neighbor.fitness_score

                # Enhanced acceptance criteria
                if neighbor_score >= current_score or neighbor_score >= B:
                    current_solution = neighbor
                    current_score = neighbor_score
                    
                    # Update best solution with elite selection
                    if current_score > best_score:
                        best_solution = copy.deepcopy(current_solution)
                        best_score = current_score
                        delta_B *= alpha  # Accelerate decay
                        B = best_score * initial_boundary_buffer  # Reset boundary relative to best
                    else:
                        delta_B *= beta   # Encourage exploration

                # Adaptive nonlinear boundary adjustment
                B = max(B - delta_B * (1 + (iterations/500)), 0)  # Faster decay acceleration
                
                # Stagnation detection and diversification
                memory_window.append(current_score)
                if len(memory_window) == memory_window.maxlen:
                    if (max(memory_window) - min(memory_window)) < improvement_threshold:
                        # Stronger diversification kick
                        current_solution = self.perturb_solution(best_solution, data)
                        current_score = current_solution.fitness_score
                        B = current_score * 1.3  # Higher boundary reset
                        delta_B *= 0.7  # More aggressive decay reduction
                        
                iterations += 1

            except Exception as e:
                print(f"Iteration {iterations} failed: {str(e)}")
                break

        # Final intensification phase using sorted initial solution
        refined_solution = self.hill_climbing_combined(
            data,
            iterations=int(max_iterations*0.15)  # Increased refinement time
        )[1]
        
        # Fallback to best solution if refinement degraded quality
        if refined_solution.fitness_score < best_score:
            return best_score, best_solution
        
        return refined_solution.fitness_score, refined_solution
    def simulated_annealing_hybrid_parallel(self, data, max_iterations=1000):
        
            #Generate initial solution using GRASP
            initial_solution = self.generate_initial_solution_grasp(data, p=0.05, max_time=5)

            # Shared dictionary and lock for process synchronization
            manager = multiprocessing.Manager()
            shared_best = manager.dict()
            shared_best["score"] = initial_solution.fitness_score
            shared_best["solution"] = initial_solution
            lock = manager.Lock()

            # Launch three paralell processes with different cooling strategies

            processes = [
                multiprocessing.Process(target=self.simulated_annealing_core_mp_optimized,
                                        args=(initial_solution, data, cooling_exponential, max_iterations, shared_best, lock, "exp")),
                multiprocessing.Process(target=self.simulated_annealing_core_mp_optimized,
                                        args=(initial_solution, data, cooling_geometric, max_iterations, shared_best, lock, "geo")),
                multiprocessing.Process(target=self.simulated_annealing_core_mp_optimized,
                                        args=(initial_solution, data, cooling_lundy_mees, max_iterations, shared_best, lock, "lundy"))
            ]

            # Start all processes

            for p in processes:
                p.start()
            
            # Wait for all to finish

            for p in processes:
                p.join()

            return shared_best["score"], shared_best["solution"]

    def validate_solution(self, solution, data):
        # Validates and corrects a solution by removing excess books from libraries that exceed scanning limits.
        id_map = {lib.id: lib for lib in data.libs}
        curr_time = 0

        # Validate each library in the signed list

        for lib_id in list(solution.signed_libraries):
            library = id_map.get(lib_id)
            if library is None:
                continue  # # Skip if ID is invalid
            
            # Calculate how many books this library can scan within the time limit

            if curr_time + library.signup_days >= data.num_days:
                max_books = 0
            else:
                time_left = data.num_days - (curr_time + library.signup_days)
                max_books = time_left * library.books_per_day
           
            # Get currently planned books
            
            scanned_list = solution.scanned_books_per_library.get(lib_id, [])
            actual_count = len(scanned_list)

            # Remove excess books if over limit

            if actual_count > max_books:
                # Remove all books
                if max_books <= 0:
        
                    removed_books = set(scanned_list)
                    solution.scanned_books_per_library.pop(lib_id, None)
                else:
                    # Remove lowest-scoring books
                    sorted_books = sorted(scanned_list, key=lambda b: data.scores[b])
                    remove_count = actual_count - max_books
                    removed_books = set(sorted_books[:remove_count])
                    kept_books = [b for b in scanned_list if b not in removed_books]
                    solution.scanned_books_per_library[lib_id] = kept_books

                # Remove from global scanned books

                solution.scanned_books.difference_update(removed_books)
             
            # Advance time with registered date of this library
            
            curr_time += library.signup_days
        return solution
    
    def simulated_annealing_core_mp_optimized(self, initial_solution, data, cooling_func, iterations, shared_best, lock, name):
        current_solution = copy.deepcopy(initial_solution)
        current_solution.calculate_fitness_score(data.scores)
        best_solution = copy.deepcopy(current_solution)
        current_temp = 1000.0

        # Operator pool and their tracking names
        
        operators = [
            self.tweak_solution_insert_library, # good for exploitation
            self.tweak_solution_swap_same_books, # good for exploitation
            self.tweak_solution_swap_signed # good for exploration
        ]
        operator_names = ["insert", "same_books", "swap_signed"]

        # Track operator performance for adaptive selection

        stats = {
            name: {"gain": 1.0, "count": 1} for name in operator_names
        }

        weights = [1.0 for _ in operators]

        for iteration in range(iterations):

            # Choose operator based on current weights

            operator = random.choices(operators, weights=weights, k=1)[0]
            op_name = operator_names[operators.index(operator)]

            try:
                new_solution = operator(copy.deepcopy(current_solution), data)
                new_solution.calculate_fitness_score(data.scores)

                delta = new_solution.fitness_score - current_solution.fitness_score
                acceptance_prob = math.exp(delta / current_temp) if delta < 0 else 1.0

                if delta > 0 or random.random() < acceptance_prob:
                    current_solution = new_solution
                    if current_solution.fitness_score > best_solution.fitness_score:
                        best_solution = copy.deepcopy(current_solution)
                    stats[op_name]["gain"] += max(0, delta)

                stats[op_name]["count"] += 1

            except Exception:
                continue

            # Update temperature
            
            current_temp = cooling_func(current_temp)

            if iteration % 100 == 0:
                with lock:
                    if best_solution.fitness_score > shared_best["score"]:
                        shared_best["score"] = best_solution.fitness_score
                        shared_best["solution"] = copy.deepcopy(best_solution)
                    else:
                        current_solution = copy.deepcopy(shared_best["solution"])
                        current_solution.calculate_fitness_score(data.scores)

                # Update operator weights based on gain per use

                weights = [
                    stats[name]["gain"] / stats[name]["count"]
                    for name in operator_names
                ]
        # Final validation
        try:
            self.validate_solution(best_solution, data)
            best_solution.calculate_fitness_score(data.scores)
        except:
            print(f"[{name.upper()}] Final solution invalid.")
            return

        with lock:
            if best_solution.fitness_score > shared_best["score"]:
                shared_best["score"] = best_solution.fitness_score
                shared_best["solution"] = best_solution
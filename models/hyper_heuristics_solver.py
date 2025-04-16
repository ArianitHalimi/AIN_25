import random
import copy
import math
import time
from collections import defaultdict
from models.library import Library
from models.solution import Solution

def current_signup_time(solution, data):
    """
    Compute the total signup time based on the order of signed libraries.
    """
    lib_dict = {lib.id: lib for lib in data.libs}
    return sum(lib_dict[lib_id].signup_days for lib_id in solution.signed_libraries)

def add_library(solution, library, data):
    """
    Incrementally adds a library to the solution.
    Computes the available scanning time after the current signup time,
    selects the best available books (not already scanned) and updates the solution.
    Returns a new solution (deep copy) with the library added.
    If the library cannot be added (because no time remains or no new books are added), returns the original solution.
    """
    new_solution = copy.deepcopy(solution)
    lib_dict = {lib.id: lib for lib in data.libs}
    curr_time = current_signup_time(new_solution, data)
    if curr_time + library.signup_days >= data.num_days:
        return new_solution  # no time left to add this library
    time_left = data.num_days - (curr_time + library.signup_days)
    max_books = time_left * library.books_per_day
    candidate_books = {book.id for book in library.books} - new_solution.scanned_books
    available_books = sorted(candidate_books, key=lambda b: -data.scores[b])[:max_books]
    if not available_books:
        return new_solution  # adding library does not contribute new books
    new_solution.signed_libraries.append(library.id)
    new_solution.scanned_books_per_library[library.id] = available_books
    new_solution.scanned_books.update(available_books)
    new_solution.calculate_fitness_score(data.scores)
    return new_solution

class HyperHeuristicSolver:
    """
    Hyper-heuristic solver for the Book Scanning problem that incrementally adds libraries using
    multiple low-level heuristics. The solver also logs the names of the heuristics used and uses
    weighted selection so that heuristics that yield improvements are more likely to be selected.
    """
    def __init__(self, time_limit=300, iterations=1000):
        self.time_limit = time_limit
        self.iterations = iterations
        # List of heuristic functions (each should add one library to the solution)
        self.heuristics = [
            self.heuristic_most_books,
            self.heuristic_lowest_coefficient,
            self.heuristic_highest_scanning_capacity,
            self.heuristic_highest_total_book_score,
            self.heuristic_unique_books,
            self.heuristic_time_aware,
            self.heuristic_hybrid_ratio,
            self.heuristic_random
        ]
        # Initial weights for each heuristic (higher weight => more likely to be selected)
        self.heuristic_weights = [1.0] * len(self.heuristics)
        # Log for the names of heuristics used (for post-run analysis)
        self.heuristic_log = []

    def generate_initial_solution(self, data):
        """
        Generates an initial (empty) solution.
        All libraries are unscheduled at the beginning.
        """
        unsigned_libs = [lib.id for lib in data.libs]
        # Start with no signed libraries, no scanned books.
        solution = Solution(signed_libs=[], 
                            unsigned_libs=unsigned_libs, 
                            scanned_books_per_library={}, 
                            scanned_books=set())
        solution.calculate_fitness_score(data.scores)
        return solution

    # --- Low-level Heuristic Moves (Incremental Additions) ---
    def heuristic_most_books(self, solution, data):
        """
        Adds the library (not already in the solution) having the most books.
        """
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries]
        if not candidates:
            return solution
        best = max(candidates, key=lambda lib: len(lib.books))
        return add_library(solution, best, data)

    def heuristic_lowest_coefficient(self, solution, data):
        """
        Adds the library with the lowest cost coefficient.
        The coefficient is defined here as:
            signup_days / (books_per_day * average_book_score)
        """
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries]
        if not candidates:
            return solution
        def coeff(lib):
            if not lib.books:
                return float('inf')
            avg_score = sum(data.scores[book.id] for book in lib.books) / len(lib.books)
            return lib.signup_days / (lib.books_per_day * avg_score) if avg_score > 0 else float('inf')
        best = min(candidates, key=coeff)
        return add_library(solution, best, data)

    def heuristic_highest_scanning_capacity(self, solution, data):
        """
        Adds the library with the highest daily scanning capacity.
        """
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries]
        if not candidates:
            return solution
        best = max(candidates, key=lambda lib: lib.books_per_day)
        return add_library(solution, best, data)

    def heuristic_highest_total_book_score(self, solution, data):
        """
        Adds the library with the highest total book score.
        """
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries]
        if not candidates:
            return solution
        def total_score(lib):
            return sum(data.scores[book.id] for book in lib.books)
        best = max(candidates, key=total_score)
        return add_library(solution, best, data)

    def heuristic_unique_books(self, solution, data):
        """
        Adds the library that has the highest count of unique books (books that appear only in that library).
        """
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries]
        if not candidates:
            return solution
        book_frequency = defaultdict(int)
        for lib in data.libs:
            for book in lib.books:
                book_frequency[book.id] += 1
        def unique_count(lib):
            return sum(1 for book in lib.books if book_frequency[book.id] == 1)
        best = max(candidates, key=unique_count)
        return add_library(solution, best, data)

    def heuristic_randomized_greedy(self, solution, data):
        """
        Among the top candidates (based on total book score), randomly selects one to add.
        """
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries]
        if not candidates:
            return solution
        sorted_candidates = sorted(candidates, key=lambda lib: sum(data.scores[book.id] for book in lib.books), reverse=True)
        # Choose from the top third (at least one candidate)
        top_count = max(1, len(sorted_candidates) // 3)
        best = random.choice(sorted_candidates[:top_count])
        return add_library(solution, best, data)

    def heuristic_time_aware(self, solution, data):
        """
        Adds a library taking into account the remaining time, preferring libraries with shorter signup days.
        """
        lib_dict = {lib.id: lib for lib in data.libs}
        curr_time = current_signup_time(solution, data)
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries 
                      and (curr_time + lib.signup_days < data.num_days)]
        if not candidates:
            return solution
        best = min(candidates, key=lambda lib: lib.signup_days)
        return add_library(solution, best, data)

    def heuristic_hybrid_ratio(self, solution, data):
        """
        Adds the library with the best hybrid ratio of (total book score / signup_days)*books_per_day.
        """
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries]
        if not candidates:
            return solution
        def hybrid(lib):
            total = sum(data.scores[book.id] for book in lib.books)
            return (total / lib.signup_days) * lib.books_per_day if lib.signup_days > 0 else 0
        best = max(candidates, key=hybrid)
        return add_library(solution, best, data)

    def heuristic_random(self, solution, data):
        """
        Adds a random library from those not yet in the solution.
        """
        candidates = [lib for lib in data.libs if lib.id not in solution.signed_libraries]
        if not candidates:
            return solution
        best = random.choice(candidates)
        return add_library(solution, best, data)

    # --- Main Hyper-Heuristic Solve Method ---
    def solve(self, data):
        """
        Runs the hyper-heuristic algorithm. Starting with an empty solution, it incrementally
        adds libraries using low-level heuristics chosen via weighted random selection.
        It logs the heuristic moves used and adapts weights based on the improvement observed.
        """
        current_solution = self.generate_initial_solution(data)
        best_solution = current_solution
        start_time = time.time()
        iter_count = 0
        self.heuristic_log = []

        while time.time() - start_time < self.time_limit and iter_count < self.iterations:
            # If no candidate remains (all libraries added or no time left), break.
            if len(current_solution.signed_libraries) == len(data.libs):
                break

            # Weighted random choice of heuristic.
            chosen_heuristic = random.choices(self.heuristics, weights=self.heuristic_weights, k=1)[0]
            candidate_solution = chosen_heuristic(current_solution, data)

            # Log the used heuristic name.
            heuristic_name = chosen_heuristic.__name__
            self.heuristic_log.append(heuristic_name)

            # Update weights adaptively: if move improved fitness, increase weight; otherwise, decrease a bit.
            if candidate_solution.fitness_score > current_solution.fitness_score:
                current_solution = candidate_solution
                idx = self.heuristics.index(chosen_heuristic)
                self.heuristic_weights[idx] += 0.1
                if current_solution.fitness_score > best_solution.fitness_score:
                    best_solution = current_solution
            else:
                current_solution = candidate_solution
                idx = self.heuristics.index(chosen_heuristic)
                self.heuristic_weights[idx] = max(0.1, self.heuristic_weights[idx] - 0.05)
            iter_count += 1

        # Output the log of heuristics used (and their frequencies)
        frequency = defaultdict(int)
        for name in self.heuristic_log:
            frequency[name] += 1
        print("Heuristic usage frequencies:")
        for name, count in frequency.items():
            print(f"  {name}: {count}")

        print("Final heuristic weights:")
        for heuristic, weight in zip([h.__name__ for h in self.heuristics], self.heuristic_weights):
            print(f"  {heuristic}: {weight:.2f}")

        return best_solution

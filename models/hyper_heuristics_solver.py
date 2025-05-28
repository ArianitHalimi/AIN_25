import random
import copy
import math
import time
from collections import defaultdict
from models.library import Library
from models.solution import Solution
from models.instance_data import InstanceData
from typing import Dict, List
import heapq

def current_signup_time(solution, lib_dict):
    """
    Compute the total signup time based on the order of signed libraries.
    """
    return sum(lib_dict[lib_id].signup_days for lib_id in solution.signed_libraries)

class HyperHeuristicSolver:
    """
    Hyper-heuristic solver for the Book Scanning problem that incrementally adds libraries using
    multiple low-level heuristics. The solver also logs the names of the heuristics used and uses
    weighted selection so that heuristics that yield improvements are more likely to be selected.
    """
    def __init__(self ,time_limit=300, iterations=10000):
        self.time_limit = time_limit
        self.iterations = iterations
        self.lib_dict = None
        self.book_frequency = defaultdict(int)
        # List of heuristic functions (each should add one library to the solution)
        self.heuristics = [
            self.heuristic_most_books,
            self.heuristic_lowest_coefficient,
            self.heuristic_highest_scanning_capacity,
            self.heuristic_highest_total_book_score,
            self.heuristic_unique_books,
            self.heuristic_time_aware,
            self.heuristic_hybrid_ratio,
            self.heuristic_highest_unscanned_score,
            self.heuristic_density,
            self.heuristic_throughput_utilization,
            self.heuristic_minimal_overlap,
        ]       
        # Log for the names of heuristics used (for post-run analysis)
        self.heuristic_log = []
        self.remaining = []
        self.blackListedLibraryIds = []

    def add_library(self, solution, library, data):
        """
            Incrementally adds a library to the solution.
            Computes the available scanning time after the current signup time,
            selects the best available books (not already scanned) and updates the solution.
            Returns a new solution (deep copy) with the library added.
            If the library cannot be added (because no time remains or no new books are added), returns the original solution.
        """
        new_solution = copy.deepcopy(solution)
       
        curr_time = current_signup_time(new_solution, self.lib_dict)
        if curr_time + library.signup_days >= data.num_days:
            self.blackListedLibraryIds.append(library.id)
            return new_solution  # no time left to add this library
        time_left = data.num_days - (curr_time + library.signup_days)
        max_books = time_left * library.books_per_day
        candidate_books = {book.id for book in library.books} - new_solution.scanned_books
        available_books = heapq.nlargest(
            max_books,
            candidate_books,
            key=lambda b: data.scores[b]
        )
        if not available_books:
            self.blackListedLibraryIds.append(library.id)
            return new_solution  # adding library does not contribute new books
        new_solution.signed_libraries.append(library.id)
        new_solution.scanned_books_per_library[library.id] = available_books
        new_solution.scanned_books.update(available_books)
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

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

    def heuristic_most_books(self, solution, data, k: int):
        """Add the k libraries with the most books."""
        top_k_libs = heapq.nlargest(k, self.remaining, key=lambda lib: len(lib.books))
        for lib in top_k_libs:
            solution = self.add_library(solution, lib, data)

        return solution

    def heuristic_lowest_coefficient(self, solution, data, k: int):
        def coeff(lib):
            if not lib.books: return float('inf')
            avg = sum(data.scores[b.id] for b in lib.books) / len(lib.books)
            return lib.signup_days / (lib.books_per_day * avg) if avg>0 else float('inf')        
        top_k_libs = heapq.nsmallest(k, self.remaining, key=coeff)
        for lib in top_k_libs:
            solution = self.add_library(solution, lib, data)

        return solution

    def heuristic_highest_scanning_capacity(self, solution, data, k: int):
        top_k_libs = heapq.nlargest(k, self.remaining, key=lambda lib: lib.books_per_day)
        for lib in top_k_libs:
            solution = self.add_library(solution, lib, data)

        return solution

    def heuristic_highest_total_book_score(self, solution, data, k: int):
        top_k_libs = heapq.nlargest(k, self.remaining, key=lambda lib: sum(data.scores[b.id] for b in lib.books))
        for lib in top_k_libs:
            solution = self.add_library(solution, lib, data)

        return solution

    def heuristic_unique_books(self, solution, data, k: int):
        top_k_libs = heapq.nlargest(k, self.remaining, key=lambda lib: sum(1 for b in lib.books))
        for lib in top_k_libs:
            solution = self.add_library(solution, lib, data)
        return solution

    def heuristic_time_aware(self, solution, data, k: int):
        top_k_libs = heapq.nsmallest(k, self.remaining, key=lambda lib: lib.signup_days)
        for lib in top_k_libs:
            solution = self.add_library(solution, lib, data)

        return solution

    def heuristic_hybrid_ratio(self, sol, data, k: int):
        top_k_libs = heapq.nlargest(k, self.remaining, key=lambda lib: (sum(data.scores[b.id] for b in lib.books)/lib.signup_days)*lib.books_per_day if lib.signup_days>0 else 0)
        for lib in top_k_libs:
            sol = self.add_library(sol, lib, data)

        return sol
    
    # Highest Unscanned Total Score
    def heuristic_highest_unscanned_score(self, sol, data, k: int):
        top_k_libs = heapq.nlargest(k, self.remaining, key=lambda lib:  sum(data.scores[b.id] for b in lib.books if b.id not in sol.scanned_books))
        for lib in top_k_libs:
            sol = self.add_library(sol, lib, data)

        return sol
    
    # Density Heuristic (score per signup day)
    def heuristic_density(self, sol, data, k: int):
        def density(lib):
            unscored = [b.id for b in lib.books if b.id not in sol.scanned_books]
            total = sum(data.scores[i] for i in unscored)
            return total / lib.signup_days
        top_k_libs = heapq.nlargest(k, self.remaining, key=density)
        for lib in top_k_libs:
            sol = self.add_library(sol, lib, data)

        return sol

    # Throughput Utilization
    def heuristic_throughput_utilization(self, sol, data, k: int):
        def util(lib):
            curr = current_signup_time(sol, self.lib_dict) + lib.signup_days
            days = max(0, data.num_days - curr)
            return min(len([b for b in lib.books if b.id not in sol.scanned_books]), days * lib.books_per_day)
        top_k_libs = heapq.nlargest(k, self.remaining, key=util)
        for lib in top_k_libs:
            sol = self.add_library(sol, lib, data)

        return sol

    # Minimal Overlap Ratio
    def heuristic_minimal_overlap(self, sol, data, k: int):
        def overlap(lib):
            total = len(lib.books)
            newb = len([b for b in lib.books if b.id not in sol.scanned_books])
            return 1 - (newb / total)
        top_k_libs = heapq.nsmallest(k, self.remaining, key=overlap)
        for lib in top_k_libs:
            sol = self.add_library(sol, lib, data)

        return sol

    # Marginal Gain Heuristic
    def heuristic_marginal_gain(self, sol, data, k: int):
        out = sol
        lib_dict = self.lib_dict
        sorted_books = self.sorted_books
        for _ in range(k):
            # build remaining IDs once
            rem_ids = [lid for lid in lib_dict if lid not in out.signed_libraries]
            if not rem_ids:
                break

            # compute current signup time once
            curr_time = current_signup_time(out, data)

            best_lib_id = None
            best_gain = -1

            for lid in rem_ids:
                lib = lib_dict[lid]
                time_left = data.num_days - (curr_time + lib.signup_days)
                if time_left <= 0:
                    continue
                capacity = time_left * lib.books_per_day

                # walk the pre-sorted list and sum up to capacity
                gain = 0
                count = 0
                for bid in sorted_books[lid]:
                    if bid not in out.scanned_books:
                        gain += data.scores[bid]
                        count += 1
                        if count >= capacity:
                            break

                if gain > best_gain:
                    best_gain = gain
                    best_lib_id = lid

            if best_lib_id is None:
                break

            # now do the real addition once
            out = self.add_library(out, lib_dict[best_lib_id], data)

        return out
    

    # --- Main Hyper-Heuristic Solve Method ---
    def solve(self, data: InstanceData):
        """
        Runs the hyper-heuristic algorithm. Starting with an empty solution, it incrementally
        adds libraries using low-level heuristics chosen via weighted random selection.
        It logs the heuristic moves used and adapts weights based on the improvement observed.
        """
        self.lib_dict = {lib.id: lib for lib in data.libs}
        for lib in data.libs:
            for b in lib.books:
                self.book_frequency[b.id] += 1
        self.sorted_books = {
        lib.id: sorted((b.id for b in lib.books),
                   key=lambda bid: -data.scores[bid])
        for lib in data.libs
        }       
        current_solution = self.generate_initial_solution(data)
        candidate_solution = current_solution

        start_time = time.time()
        iter_count = 0
        self.heuristic_log = []
        heuristic_log_index = []
        maxk = 9

        while iter_count < self.iterations:
            if len(current_solution.signed_libraries) == len(data.libs):
                print("All librairies signed")
                break    
            
            if len(self.blackListedLibraryIds) == len(current_solution.unsigned_libraries):
                print("Libraries blacklisted")
                break

            self.remaining = [lib for lib in data.libs if lib.id not in candidate_solution.signed_libraries]
            # Generate a random k
            k = random.randint(1, maxk)
            # Greedy approach: pick the heuristic that gives the best result
            chosen_heuristic= None
            for heuristic in self.heuristics:
                # try to use each one and pick the best
                sub_candidate_solution = heuristic(current_solution, data, k)
                if(sub_candidate_solution.fitness_score > candidate_solution.fitness_score):
                    candidate_solution = sub_candidate_solution
                    chosen_heuristic = heuristic
        
            current_solution = candidate_solution
            if(chosen_heuristic is None and maxk == 1): 
                print("Heuristics did not improve ")
                break
            if(chosen_heuristic is None):
                print("set maxk to 1")
                maxk = 1
                continue
            idx = self.heuristics.index(chosen_heuristic)
             # Log the used heuristic name.
            heuristic_name = chosen_heuristic.__name__
            self.heuristic_log.append(heuristic_name)
            heuristic_log_index.append(idx)
            iter_count += 1


        # while iter_count < self.iterations:
        #     # If no candidate remains (all libraries added or no time left), break.
        #     if len(current_solution.signed_libraries) == len(data.libs):
        #         break


            
        #     # Weighted random choice of heuristic.
        #     chosen_heuristic = random.choices(self.heuristics, weights=self.heuristic_weights, k=1)[0]
        #     candidate_solution = chosen_heuristic(current_solution, data)
        #     idx = self.heuristics.index(chosen_heuristic)

        #     # Log the used heuristic name.
        #     heuristic_name = chosen_heuristic.__name__
        #     self.heuristic_log.append(heuristic_name)
        #     heuristic_log_index.append(idx)
        #     current_solution = candidate_solution
        #     iter_count += 1

        # Output the log of heuristics used (and their frequencies)
        frequency = defaultdict(int)

        for name in self.heuristic_log:
            frequency[name] += 1
        
        print(f"total iterations: {len(heuristic_log_index)}")

        return (current_solution, heuristic_log_index)

 # python validator.py ../input/B90000_L850_D21.txt ../output/hyper/B90000_L850_D21/B90000_L850_D21.txt
 # python validator.py ../input/UPFIEK.txt ../output/hyper/UPFIEK/UPFIEK.txt
 # python validator.py ../input/switch_books_instance.txt ../output/hyper/switch_books_instance/switch_books_instance.txt
 # python validator.py ../input/f_libraries_of_the_world.txt ../output/hyper/f_libraries_of_the_world/f_libraries_of_the_world.txt
 # python validator.py ../input/e_so_many_books.txt ../output/hyper/e_so_many_books/e_so_many_books.txt
 # python validator.py ../input/d_tough_choices.txt ../output/hyper/d_tough_choices/d_tough_choices.txt
 # python validator.py ../input/c_incunabula.txt ../output/hyper/c_incunabula/c_incunabula.txt
 # python validator.py ../input/b_read_on.txt ../output/hyper/b_read_on/b_read_on.txt
# python validator.py ../input/B5000_L90_D21.txt ../output/hyper/B5000_L90_D21/B5000_L90_D21.txt
# python validator.py ../input/B50000_L400_D28.txt ../output/hyper/B50000_L400_D28/B50000_L400_D28.txt
# python validator.py ../input/B95000_L2000_D28.txt ../output/hyper/B95000_L2000_D28/B95000_L2000_D28.txt
# python validator.py ../input/B100000_L600_D28.txt ../output/hyper/B100000_L600_D28/B100000_L600_D28.txt
# python validator.py ../input/a_example.txt ../output/hyper/a_example/a_example.txt
# python validator.py ../input/B100_L9_D18.in ../output/hyper/B100_L9_D18/B100_L9_D18.in
# python validator.py ../input/B70_L8_D7.in ../output/hyper/B70_L8_D7/B70_L8_D7.in
# python validator.py ../input/B95_L5_D12.in ../output/hyper/B95_L5_D12/B95_L5_D12.in

# # python validator.py ../input/e_so_many_books.txt ../output/hyper/e_so_many_books/aco_e_so_many_books.txt



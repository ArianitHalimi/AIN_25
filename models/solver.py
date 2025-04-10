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
        book_count = defaultdict(int)
        unscanned_books_per_library = {}

        for library in data.libs:
            if library.id in solution.signed_libraries:
                unsigned_books = []
                for book in library.books:
                    book_count[book.id] += 1
                    if book.id not in solution.scanned_books_per_library.get(library.id,
                                                                             []) and book.id not in solution.scanned_books:
                        unsigned_books.append(book.id)
                if len(unsigned_books) > 0:
                    unscanned_books_per_library[library.id] = unsigned_books

        if len(unscanned_books_per_library) == 1:
            # print("Only 1 library with unscanned books was found")
            return solution

        possible_books = [
            book_id for book_id, count in book_count.items()
            if count > 1 and book_id in solution.scanned_books
        ]

        valid_books = set()

        for library, unscanned_books in unscanned_books_per_library.items():
            for book_id in possible_books:
                for book in data.libs[library].books:
                    if book.id == book_id:
                        valid_books.add(book_id)

        if not valid_books:
            # print("No valid books were found")
            return solution  # No book meets the criteria, return unchanged

        # Get random book to swap
        book_to_move = random.choice(list(valid_books))

        # Identify which library is currently scanning this book
        current_library = None
        for lib_id, books in solution.scanned_books_per_library.items():
            if book_to_move in books:
                current_library = lib_id
                break

        if unscanned_books_per_library.get(current_library) is None or len(
                unscanned_books_per_library[current_library]) == 0:
            return solution

        # Select other library with any un-scanned books to scan this book
        possible_libraries = [
            lib for lib in data.book_libs[book_to_move]
            if lib != current_library and any(
                library.id == lib and any(book.id not in solution.scanned_books for book in library.books)
                for library in data.libs if library.id in solution.signed_libraries
            )
        ]

        if len(possible_libraries) == 0:
            print("No valid libraries were found")
            return solution

        new_library = random.choice(possible_libraries)

        # Remove the book from the current library
        solution.scanned_books_per_library[current_library].remove(book_to_move)
        solution.scanned_books.remove(book_to_move)

        # Add the book to the new library, maintaining feasibility
        current_books_in_new_library = solution.scanned_books_per_library[new_library]

        # Ensure feasibility: If new_library is at its limit, remove a book to make space
        max_books_per_day = data.libs[new_library].books_per_day

        days_before_sign_up = 0
        found = False

        for id in solution.signed_libraries:
            if found:
                break
            days_before_sign_up += data.libs[id].signup_days
            if id == new_library:
                found = True

        numOfDaysAvailable = data.num_days - days_before_sign_up

        book_to_remove = None
        if len(current_books_in_new_library) > numOfDaysAvailable * max_books_per_day:
            book_to_remove = random.choice(list(current_books_in_new_library))
            current_books_in_new_library.remove(book_to_remove)
            solution.scanned_books.remove(book_to_remove)

        # Add the book to the new library
        current_books_in_new_library.append(book_to_move)
        solution.scanned_books.add(book_to_move)

        books_in_current_library = solution.scanned_books_per_library[current_library]

        new_scanned_book = random.choice(unscanned_books_per_library.get(current_library))
        books_in_current_library.append(new_scanned_book)
        solution.scanned_books.add(new_scanned_book)

        solution.calculate_delta_fitness(data, new_scanned_book, book_to_remove)

        return solution

    def hill_climbing_swap_signed(self, data, iterations = 1000):
        solution = self.generate_initial_solution(data)
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
        solution = self.generate_initial_solution(data)

        for i in range(iterations - 1):
            new_solution = self.tweak_solution_swap_signed_with_unsigned(solution, data)
            # new_solution = self.tweak_solution_signed_unsigned(solution, data, bias_type="favor_second_half")
            # new_solution = self.tweak_solution_signed_unsigned(solution, data, bias_type="favor_first_half", bias_ratio=3/4)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return (solution.fitness_score, solution)

    def random_search(self, data, iterations = 1000):
        solution = self.generate_initial_solution(data)

        for i in range(iterations - 1):
            new_solution = self.generate_initial_solution(data)

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
        solution = self.generate_initial_solution(data)

        for i in range(iterations):
            new_solution = self.tweak_solution_swap_same_books(solution, data)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return (solution.fitness_score, solution)

    def hill_climbing_combined(self, data, iterations = 1000):
        solution = self.generate_initial_solution(data)

        list_of_climbs = [
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_same_books,
            self.tweak_solution_swap_signed,
            self.tweak_solution_swap_last_book
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
        solution = self.generate_initial_solution(data)

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

        Library._id_counter = 0 
        S = self.generate_initial_solution(data)
        print(f"Initial solution fitness: {S.fitness_score}")

        H = copy.deepcopy(S)
        Best = copy.deepcopy(S)

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

                R = self.tweak_solution_swap_signed_with_unsigned(
                    copy.deepcopy(S), data
                )

                if R.fitness_score > S.fitness_score:
                    S = copy.deepcopy(R)
                    
                if S.fitness_score >= data.calculate_upper_bound():
                    return (S.fitness_score, S)

                total_iterations += 1
                if total_iterations >= max_iterations:
                    break

            if S.fitness_score > Best.fitness_score:
                Best = copy.deepcopy(S)

            if S.fitness_score >= H.fitness_score:
                H = copy.deepcopy(S)
            else:

                if random.random() < 0.1:
                    H = copy.deepcopy(S)

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
            min(1, max_destroy_size),
            min(max_destroy_size, max_destroy_size // 3 + 1)
        )
        
        libraries_to_remove = random.sample(perturbed.signed_libraries, destroy_size)
        
        new_signed_libraries = [lib for lib in perturbed.signed_libraries if lib not in libraries_to_remove]
        new_unsigned_libraries = perturbed.unsigned_libraries + libraries_to_remove
        
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        
        for lib_id in new_signed_libraries:
            if lib_id in perturbed.scanned_books_per_library:
                new_scanned_books_per_library[lib_id] = perturbed.scanned_books_per_library[lib_id].copy()
                new_scanned_books.update(new_scanned_books_per_library[lib_id])
        
        curr_time = sum(data.libs[lib_id].signup_days for lib_id in new_signed_libraries)
        
        lib_scores = []
        for lib_id in new_unsigned_libraries:
            library = data.libs[lib_id]
            available_books = [b for b in library.books if b.id not in new_scanned_books]
            if not available_books:
                continue
            avg_score = sum(data.scores[b.id] for b in available_books) / len(available_books)
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
                key=lambda b: -data.scores[b]
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
            new_scanned_books
        )
        rebuilt_solution.calculate_fitness_score(data.scores)
        
        return rebuilt_solution

    def hill_climbing_insert_library(self, data, iterations=1000):
        Library._id_counter = 0
        
        # 1. Preprocess and validate data
        valid_library_ids = set(range(len(data.libs)))

        # Convert book_libs to dictionary format if it's a list
        if isinstance(data.book_libs, list):
            # Assuming list index corresponds to book_id
            book_libs_dict = {}
            for book_id, lib_ids in enumerate(data.book_libs):
                if isinstance(lib_ids, (list, tuple)):
                    book_libs_dict[book_id] = [
                        lib_id for lib_id in lib_ids
                        if lib_id in valid_library_ids
                    ]
            data.book_libs = book_libs_dict
        elif hasattr(data, 'book_libs') and isinstance(data.book_libs, dict):
            # Clean existing dictionary
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

        # Rest of the function remains the same as previous solution
        solution = self.generate_initial_solution(data)
        solution.unsigned_libraries = [
            lib_id for lib_id in solution.unsigned_libraries
            if lib_id in valid_library_ids
        ]

        for _ in range(iterations):
            # Get unsigned books from valid libraries
            unsigned_books = []
            for lib_id in solution.unsigned_libraries:
                if lib_id not in valid_library_ids:
                    continue
                lib = data.libs[lib_id]
                for book in lib.books:
                    if book.id not in solution.scanned_books:
                        unsigned_books.append(book)

            if not unsigned_books:
                continue

            # Select top book
            unsigned_books.sort(key=lambda b: -data.scores[b.id])
            selected_book = random.choice(unsigned_books[:max(1, len(unsigned_books)//20)])

            # Find valid libraries containing this book
            if selected_book.id not in data.book_libs:
                continue

            library_candidates = [
                lib_id for lib_id in data.book_libs[selected_book.id]
                if lib_id in valid_library_ids and
                   lib_id in solution.unsigned_libraries
            ]

            if not library_candidates:
                continue

            # Sign the library
            selected_library = random.choice(library_candidates)
            solution.signed_libraries.append(selected_library)
            solution.unsigned_libraries.remove(selected_library)

            # Calculate available scanning time
            total_signup = sum(
                data.libs[lib_id].signup_days
                for lib_id in solution.signed_libraries
                if lib_id in valid_library_ids
            )
            time_left = data.num_days - total_signup

            if time_left <= 0:
                solution.signed_libraries.pop()
                solution.unsigned_libraries.append(selected_library)
                continue

            # Scan books
            lib = data.libs[selected_library]
            available_books = sorted(
                {b.id for b in lib.books} - solution.scanned_books,
                key=lambda b: -data.scores[b]
            )[:time_left * lib.books_per_day]

            if available_books:
                solution.scanned_books_per_library[selected_library] = available_books
                solution.scanned_books.update(available_books)

            # Ensure time constraint
            while True:
                total_signup = sum(
                    data.libs[lib_id].signup_days
                    for lib_id in solution.signed_libraries
                    if lib_id in valid_library_ids
                )
                if total_signup <= data.num_days:
                    break

                if not solution.signed_libraries:
                    break

                removed = solution.signed_libraries.pop()
                solution.unsigned_libraries.append(removed)
                if removed in solution.scanned_books_per_library:
                    solution.scanned_books.difference_update(
                        solution.scanned_books_per_library.pop(removed)
                    )

            solution.calculate_fitness_score(data.scores)

        return solution.fitness_score, solution

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
        current = create_light_solution(self.generate_initial_solution(data))
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
            current = create_light_solution(self.generate_initial_solution(data))
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

    def steepest_ascent_hill_climbing(self, data, total_time_ms=1000, n=5):
        start_time = time.time() * 1000
        current_solution = self.generate_initial_solution(data)
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
                current_solution = self.generate_initial_solution(data)
                
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
        current_solution = self.generate_initial_solution(data)
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
    

    def hill_climbing_insert_library(self, data, iterations=1000):
            # Reset the Library ID counter to ensure we start from 0
            Library._id_counter = 0

            # 1. Preprocess and validate data
            valid_library_ids = set(range(len(data.libs)))

            # Convert book_libs to dictionary format if it's a list
            if isinstance(data.book_libs, list):
                # Assuming list index corresponds to book_id
                book_libs_dict = {}
                for book_id, lib_ids in enumerate(data.book_libs):
                    if isinstance(lib_ids, (list, tuple)):
                        book_libs_dict[book_id] = [
                            lib_id for lib_id in lib_ids
                            if lib_id in valid_library_ids
                        ]
                data.book_libs = book_libs_dict
            elif hasattr(data, 'book_libs') and isinstance(data.book_libs, dict):
                # Clean existing dictionary
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

            # Rest of the function remains the same as previous solution
            solution = self.generate_initial_solution(data)
            solution.unsigned_libraries = [
                lib_id for lib_id in solution.unsigned_libraries
                if lib_id in valid_library_ids
            ]

            for _ in range(iterations):
                # Get unsigned books from valid libraries
                unsigned_books = []
                for lib_id in solution.unsigned_libraries:
                    if lib_id not in valid_library_ids:
                        continue
                    lib = data.libs[lib_id]
                    for book in lib.books:
                        if book.id not in solution.scanned_books:
                            unsigned_books.append(book)

                if not unsigned_books:
                    continue

                # Select top book
                unsigned_books.sort(key=lambda b: -data.scores[b.id])
                selected_book = random.choice(unsigned_books[:max(1, len(unsigned_books)//20)])

                # Find valid libraries containing this book
                if selected_book.id not in data.book_libs:
                    continue

                library_candidates = [
                    lib_id for lib_id in data.book_libs[selected_book.id]
                    if lib_id in valid_library_ids and
                       lib_id in solution.unsigned_libraries
                ]

                if not library_candidates:
                    continue

                # Sign the library
                selected_library = random.choice(library_candidates)
                solution.signed_libraries.append(selected_library)
                solution.unsigned_libraries.remove(selected_library)

                # Calculate available scanning time
                total_signup = sum(
                    data.libs[lib_id].signup_days
                    for lib_id in solution.signed_libraries
                    if lib_id in valid_library_ids
                )
                time_left = data.num_days - total_signup

                if time_left <= 0:
                    solution.signed_libraries.pop()
                    solution.unsigned_libraries.append(selected_library)
                    continue

                # Scan books
                lib = data.libs[selected_library]
                available_books = sorted(
                    {b.id for b in lib.books} - solution.scanned_books,
                    key=lambda b: -data.scores[b]
                )[:time_left * lib.books_per_day]

                if available_books:
                    solution.scanned_books_per_library[selected_library] = available_books
                    solution.scanned_books.update(available_books)

                # Ensure time constraint
                while True:
                    total_signup = sum(
                        data.libs[lib_id].signup_days
                        for lib_id in solution.signed_libraries
                        if lib_id in valid_library_ids
                    )
                    if total_signup <= data.num_days:
                        break

                    if not solution.signed_libraries:
                        break

                    removed = solution.signed_libraries.pop()
                    solution.unsigned_libraries.append(removed)
                    if removed in solution.scanned_books_per_library:
                        solution.scanned_books.difference_update(
                            solution.scanned_books_per_library.pop(removed)
                        )

                solution.calculate_fitness_score(data.scores)

            return solution.fitness_score, solution
    
    def guided_local_search(self, data, max_time=300, max_iterations=1000):
        C = set(range(len(data.libs)))  # Set of all possible library indices
        
        T = list(range(5, 31, 5))  # Time intervals from 5 to 30 seconds in steps of 5
        
        p = [0] * len(data.libs)
        
        S = self.generate_initial_solution(data)
        
        Best = copy.deepcopy(S)
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < max_time and iteration < max_iterations:
            local_time_limit = time.time() + random.choice(T)
            
            while time.time() < local_time_limit and time.time() - start_time < max_time:
                R = self.tweak_solution_swap_signed_with_unsigned(copy.deepcopy(S), data)
                
                if R.fitness_score > Best.fitness_score:
                    Best = copy.deepcopy(R)
                    print(f"New best score: {Best.fitness_score:,} at iteration {iteration}")
                
                R_quality = R.fitness_score - sum(p[i] for i in R.signed_libraries if i in C)
                S_quality = S.fitness_score - sum(p[i] for i in S.signed_libraries if i in C)
                if R_quality > S_quality:
                    S = copy.deepcopy(R)
                
                if Best.fitness_score >= data.calculate_upper_bound():
                    return Best
            
            C_prime = set()
            
            for Ci in S.signed_libraries:
                if Ci not in C:
                    continue
                    
                is_most_penalizable = True
                Ci_utility = sum(data.scores[book.id] for book in data.libs[Ci].books)
                Ci_penalizability = Ci_utility / (1 + p[Ci])
                
                for Cj in S.signed_libraries:
                    if Cj == Ci or Cj not in C:
                        continue
                    Cj_utility = sum(data.scores[book.id] for book in data.libs[Cj].books)
                    Cj_penalizability = Cj_utility / (1 + p[Cj])
                    if Cj_penalizability > Ci_penalizability:
                        is_most_penalizable = False
                        break
                
                if is_most_penalizable:
                    C_prime.add(Ci)
            
            for Ci in S.signed_libraries:
                if Ci in C_prime:
                    p[Ci] += 1
            
            iteration += 1
            
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration}, Time: {elapsed:.2f}s, Best score: {Best.fitness_score:,}")
        
        print(f"Search completed after {iteration} iterations and {time.time() - start_time:.2f} seconds")
        return Best

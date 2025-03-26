import random
from collections import defaultdict
from models.library import Library
import os
# from tqdm import tqdm
from models.solution import Solution
import copy
import random

class Solver:
    def __init__(self):
        pass

    def generate_initial_solution(self, data):
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
                unsigned_libraries.append(f"Library {library.id}")
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
            print("Only 1 library with unscanned books was found")
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
            print("No valid books were found")
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
        # um doket init duhet me kan copy tani qka bohet shum copied 
        # jo, kan than qe tweak osht inplace, dmth duhet me ju dergu copy
        # po a , dam memory leak 
        for i in range(iterations):
            solution_clone = copy.deepcopy(solution) #HOW??
            # improt copy lib koka pa build in prit
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
        local_signed_libs[signed_idx] = f"Library {unsigned_lib_id}"
        local_unsigned_libs[unsigned_idx] = f"Library {signed_lib_id}"

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
                solution.unsigned_libraries.append(f"Library {library.id}")
                continue

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_signed_libraries.append(f"Library {library.id}")
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
        solution = self.generate_initial_solution()

        list_of_climbs = [
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_same_books,
            self.tweak_solution_swap_signed,
        ]
        
        for _ in range(iterations - 1):
            target_climb = random.choice(list_of_climbs)

            solution_copy = copy.deepcopy(solution)
            new_solution = target_climb(solution_copy, data) 

            if (new_solution[0] > solution.fitness_score):
                solution = new_solution

        return (solution.fitness_score, solution)
import random
from collections import defaultdict

from tqdm import tqdm

from models.solution import Solution


class Solver:
    def __init__(self):
        pass

    # def solve(self, data):
    # random.shuffle(data.libs)

    # for library in tqdm(data.libs):
    #     if self.curr_time + library.signup_days >= data.num_days:
    #         self.unsigned_libraries.append(f"Library {library.id}")
    #         continue

    #     time_left = data.num_days - (self.curr_time + library.signup_days)
    #     max_books_scanned = time_left * library.books_per_day

    #     available_books = sorted(
    #         set(library.books) - self.assigned_books, key=lambda b: -data.scores[b]
    #     )[:max_books_scanned]

    #     if available_books:
    #         self.signed_libraries.append(f"Library {library.id}")
    #         self.scanned_books[library.id] = available_books
    #         self.assigned_books.update(available_books)
    #         self.curr_time += library.signup_days

    # all_books = set(range(data.num_books))
    # scanned_books_list = sorted(self.assigned_books)
    # not_scanned_books_list = sorted(all_books - self.assigned_books)

    # os.makedirs(os.path.dirname(self.libraries_output_file_name), exist_ok=True)

    # with open(self.output_file_name, "w+") as lofp:
    #     lofp.write("Signed libraries: " + ", ".join(self.signed_libraries) + "\n")
    #     lofp.write("Unsigned libraries: " + ", ".join(self.unsigned_libraries) + "\n")
    #     lofp.write("\nScanned books per library:\n")
    #     for library_id, books in self.scanned_books.items():
    #         lofp.write(f"Library {library_id}: " + ", ".join(map(str, books)) + "\n")
    #     lofp.write("\nOverall scanned books: " + ", ".join(map(str, scanned_books_list)) + "\n")
    #     lofp.write("Not scanned books: " + ", ".join(map(str, not_scanned_books_list)) + "\n")

    # print(f"Signed libraries: {len(self.signed_libraries)}")
    # print(f"Unsigned libraries: {len(self.unsigned_libraries)}")
    # print(f"Scanned books: {len(scanned_books_list)}")
    # print(f"Not scanned books: {len(not_scanned_books_list)}")

    # with open(f"{self.libraries_output_file_name}", "w+") as ofp:
    #     ofp.write(f"{len(self.signed_libraries)}\n")
    #     for library in self.signed_libraries:
    #         library_idx = int(library.split()[-1])
    #         books = self.scanned_books.get(library_idx, [])
    #         ofp.write(f"{library_idx} {len(books)}\n")
    #         ofp.write(" ".join(map(str, books)) + "\n")

    # print(f"Processing complete! Output written to: {self.output_file_name}")
    # print(f"Libraries summary saved to: {self.libraries_output_file_name}")

    def generateInitialSolution(self, data):

        shuffled_libs = data.libs.copy()
        random.shuffle(shuffled_libs)

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for library in tqdm(shuffled_libs):
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(f"Library {library.id}")
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books, key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)  # .append(f"Library {library.id}")
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        solution = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)
        solution.calculate_fitness_score(data.scores)
        print("Solution fitness score score: ", solution.fitness_score)
        return solution

    def tweak_solution(self, solution, data):
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

        print(f'Fitness before tweaking: {solution.fitness_score}')
        solution.calculate_delta_fitness(data, new_scanned_book, book_to_remove)
        print(f'Fitness after tweaking: {solution.fitness_score}')

        return solution

    def hill_climbing(self, data):
        solution = self.generateInitialSolution(data)

        for i in range(100):
            new_solution = self.tweak_solution(solution, data)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return solution

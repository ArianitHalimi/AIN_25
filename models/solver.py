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
        """
        Tweaks the given solution by swapping a book between libraries where possible.

        :param solution: The current solution object.
        :param data: The instance data containing information about libraries and books.
        :return: A new solution object with the tweak applied.
        """
        # Choose a random book that has been scanned
        if not solution.scanned_books:
            return solution  # No scanned books to tweak

        book_count = defaultdict(int)

        # Iterate through each library and count occurrences of book IDs
        for library in data.libs:
            unique_books = {book.id for book in library.books if library.id in solution.signed_libraries}
            for book_id in unique_books:
                book_count[book_id] += 1

        possible_books = [
            book_id for book_id, count in book_count.items()
            if count > 1 and book_id in solution.scanned_books
        ]

        # todo filter book here
        valid_books = [
            # book_id for book_id in possible_books
            # if any(
            #     any(book.id not in solution.scanned_books for book in data.libs[scanning_library].books)
            #     # Library has unscanned books
            #     for scanning_library, books in solution.scanned_books_per_library.items()
            #     # Find which library scanned this book
            #     if book_id in books  # Ensure this book was scanned by this library
            # )
        ]

        if not valid_books:
            return solution  # No book meets the criteria, return unchanged

        # Get random book to swap
        book_to_move = random.choice(valid_books)

        # Identify which library is currently scanning this book
        current_library = None
        for lib_id, books in solution.scanned_books_per_library.items():
            if book_to_move in books:
                current_library = lib_id
                break

        # Select other library with any un-scanned books to scan this book
        new_library = random.choice([
            lib for lib in data.book_libs[book_to_move]
            if lib != current_library and any(
                library.id == lib and any(book.id not in solution.scanned_books for book in library.books)
                for library in data.libs
            )
        ])

        if new_library is None:
            return solution

        # Remove the book from the current library
        solution.scanned_books_per_library[current_library].remove(book_to_move)
        solution.scanned_books.remove(book_to_move)

        # Add the book to the new library, maintaining feasibility
        current_books_in_new_library = solution.scanned_books_per_library[new_library]

        # Ensure feasibility: If new_library is at its limit, remove a book to make space
        max_books_per_day = data.libs[new_library].books_per_day

        # todo qiky kushti so mire
        if int(len(current_books_in_new_library) / max_books_per_day) >= (data.num_days - new_library.signup_days):
            book_to_remove = random.choice(list(current_books_in_new_library))
            current_books_in_new_library.remove(book_to_remove)
            solution.scanned_books.remove(book_to_remove)

        # Add the book to the new library
        current_books_in_new_library.append(book_to_move)
        solution.scanned_books.add(book_to_move)

        books_in_current_library = solution.scanned_books_per_library[current_library]
        current_library_unscanned_books = [book for book in books_in_current_library if
                                           book not in solution.scanned_books]
        book_to_scan = random.choice(current_library_unscanned_books)

        books_in_current_library.append(book_to_scan)
        solution.scanned_books.add(book_to_scan)

        return solution

    def hill_climbing(self, data):
        solution = self.generateInitialSolution(data)

        for i in range(100):
            new_solution = self.tweak_solution(solution, data)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return solution

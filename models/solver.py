import random
# from tqdm import tqdm
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
                signed_libraries.append(f"Library {library.id}")
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        solution = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)

        solution.calculate_fitness_score(data.scores)

        return solution

    def tweak_solution(self, solution, data):
        return solution
    
    def hill_climbing(self, data):
        solution = self.generateInitialSolution(data)

        for i in range(100):
            new_solution = self.tweak_solution(solution, data)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return solution.fitness_score, solution

    # region Hill Climbing Signed & Unsigned libs
    def _extract_lib_id(self, libraries, library_index):
        return int(libraries[library_index][len("Library "):])

    def tweak_solution_signed_unsigned(self, solution, data, bias_type=None, bias_ratio=2/3):
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

        signed_lib_id = self._extract_lib_id(local_signed_libs, signed_idx)
        unsigned_lib_id = self._extract_lib_id(local_unsigned_libs, unsigned_idx)

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
            lib_id = self._extract_lib_id(solution.signed_libraries, i)
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
            lib_id = self._extract_lib_id(local_signed_libs, i)
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

    def hill_climbing_signed_unsigned(self, data, iterations=1000):
        solution = self.generateInitialSolution(data)

        for i in range(iterations - 1):
            new_solution = self.tweak_solution_signed_unsigned(solution, data)
            # new_solution = self.tweak_solution_signed_unsigned(solution, data, bias_type="favor_second_half")
            # new_solution = self.tweak_solution_signed_unsigned(solution, data, bias_type="favor_first_half", bias_ratio=3/4)

            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution

        return solution
    # endregion

    def random_search(self, data, iterations = 1000):
        solution = self.generateInitialSolution(data)
        fitness_score = solution.fitness_score

        for i in range(iterations - 1):
            new_solution = self.generateInitialSolution(data)

            if new_solution.fitness_score > fitness_score:
                solution = new_solution
                fitness_score = new_solution.fitness_score

        return (fitness_score, solution)
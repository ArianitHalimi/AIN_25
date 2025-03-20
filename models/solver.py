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
        pass
    
    def hill_climbing(self, data):
        solution = self.generateInitialSolution(data)

        for i in range(100):
            new_solution = self.tweak_solution(solution, data)

            if new_solution.fitness_score() > solution.fitness_score():
                solution = new_solution

        return solution
    
    def tweak_solution_1(self, solution, data):

        library_ids = []
        for lib in solution.signed_libraries:
            lib_id = int(lib.split()[-1])
            if lib_id < len(data.libs):
                library_ids.append(lib_id)
        
        if len(library_ids) < 2:
            return solution
            
        idx1 = random.randint(0, len(library_ids) - 1)
        idx2 = random.randint(0, len(library_ids) - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, len(library_ids) - 1)
            
        library_ids[idx1], library_ids[idx2] = library_ids[idx2], library_ids[idx1]
        
        ordered_libs = []
        for lib_id in library_ids:
            ordered_libs.append(data.libs[lib_id])
        
        all_lib_ids = set(range(len(data.libs)))
        remaining_lib_ids = all_lib_ids - set(library_ids)
        for lib_id in remaining_lib_ids:
            ordered_libs.append(data.libs[lib_id])
            
        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for library in ordered_libs:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(f"Library {library.id}")
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books, 
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(f"Library {library.id}")
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        new_solution = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)
        new_solution.calculate_fitness_score(data.scores)
        
        return new_solution
    
    def hill_climbing_1(self, data):
        solution = self.generateInitialSolution(data)
        current_fitness = solution.fitness_score
        print("Current fitness score:", current_fitness)

        for i in range(1000):
            new_solution = self.tweak_solution_1(solution, data)
            
            if new_solution.fitness_score > current_fitness:
                solution = new_solution
                current_fitness = new_solution.fitness_score

        return (current_fitness, solution)
    
    def random_search(self, data, iterations = 1000):
        solution = self.generateInitialSolution(data)
        fitness_score = solution.fitness_score

        for i in range(iterations - 1):
            new_solution = self.generateInitialSolution(data)

            if new_solution.fitness_score > fitness_score:
                solution = new_solution
                fitness_score = new_solution.fitness_score

        return (fitness_score, solution)
import sys
import random
from tqdm import tqdm
from models import Parser

class Solution:
    def __init__(self, signed_libraries, unsigned_libraries, scanned_books, assigned_books, total_books):
        self.signed_libraries = signed_libraries
        self.unsigned_libraries = unsigned_libraries
        self.scanned_books = scanned_books
        self.assigned_books = assigned_books
        self.total_books = total_books

    def write_results(self):
        libraries_output_file = 'libraries_output.txt'
        with open(libraries_output_file, "w+") as lofp:
            lofp.write("Signed libraries: " + ", ".join(self.signed_libraries) + "\n")
            lofp.write("Unsigned libraries: " + ", ".join(self.unsigned_libraries) + "\n")
            lofp.write("\nScanned books per library:\n")
            for library_id, books in self.scanned_books.items():
                lofp.write(f"Library {library_id}: " + ", ".join(map(str, books)) + "\n")
            lofp.write("\nOverall scanned books: " + ", ".join(map(str, sorted(self.assigned_books))) + "\n")
            lofp.write("Not scanned books: " + ", ".join(map(str, sorted(self.total_books - self.assigned_books))) + "\n")

        output_file = 'result.txt'
        with open(output_file, "w+") as ofp:
            ofp.write(f"{len(self.signed_libraries)}\n")
            for library in self.signed_libraries:
                library_idx = int(library.split()[-1])
                books = self.scanned_books.get(library_idx, [])
                ofp.write(f"{library_idx} {len(books)}\n")
                ofp.write(" ".join(map(str, books)) + "\n")

        print(f"Processing complete! Output written to: {output_file}")
        print(f"Libraries summary saved to: {libraries_output_file}")


    @staticmethod
    def generateInitialSolution(input_file):
        parser = Parser(input_file)
        data = parser.parse()
        data.describe()
        
        random.shuffle(data.libs)

        signed_libraries = []
        unsigned_libraries = []
        scanned_books = {}
        assigned_books = set()
        curr_time = 0

        for library in tqdm(data.libs):
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(f"Library {library.id}")
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                set(library.books) - assigned_books, key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(f"Library {library.id}")
                scanned_books[library.id] = available_books
                assigned_books.update(available_books)
                curr_time += library.signup_days

        all_books = set(range(data.num_books))
        solution = Solution(signed_libraries, unsigned_libraries, scanned_books, assigned_books, all_books)
        solution.write_results()
        return solution
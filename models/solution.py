import random
from tqdm import tqdm

class Solution:
    signed_libraries = []
    unsigned_libraries = []
    scanned_books_per_library = {}
    scanned_books = set()

    def __init__(self, signed_libs, unsigned_libs,scanned_books_per_library, scanned_books):
        self.signed_libraries = signed_libs
        self.unsigned_libraries = unsigned_libs
        self.scanned_books_per_library = scanned_books_per_library
        self.scanned_books = scanned_books

    def export(self, file_path):
        with open(file_path, "w+") as ofp:
            ofp.write(f"{len(self.signed_libraries)}\n")
            for library in self.signed_libraries:
                library_idx = int(library.split()[-1])
                books = self.scanned_books_per_library.get(library_idx, [])
                ofp.write(f"{library_idx} {len(books)}\n")
                ofp.write(" ".join(map(str, books)) + "\n")

        print(f"Processing complete! Output written to: {file_path}")

    def describe(self, file_path="./output/output.txt"):
        with open(file_path, "w+") as lofp:
            lofp.write("Signed libraries: " + ", ".join(self.signed_libraries) + "\n")
            lofp.write("Unsigned libraries: " + ", ".join(self.unsigned_libraries) + "\n")
            lofp.write("\nScanned books per library:\n")
            for library_id, books in self.scanned_books_per_library.items():
                lofp.write(f"Library {library_id}: " + ", ".join(map(str, books)) + "\n")
            lofp.write("\nOverall scanned books: " + ", ".join(map(str, sorted(self.scanned_books))) + "\n")
            lofp.write("Not scanned books: " + ", ".join(map(str, sorted(self.total_books - self.scanned_books))) + "\n")

    def fitness_score(self):
        pass

    @staticmethod
    def generateInitialSolution(data):
       
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
                signed_libraries.append(f"Library {library.id}")
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        return Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)
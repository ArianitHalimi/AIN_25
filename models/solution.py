class Solution:
    signed_libraries = []
    unsigned_libraries = []
    scanned_books_per_library = {}
    scanned_books = set()
    fitness_score = -1

    def __init__(self, signed_libs, unsigned_libs, scanned_books_per_library, scanned_books):
        self.signed_libraries = signed_libs
        self.unsigned_libraries = unsigned_libs
        self.scanned_books_per_library = scanned_books_per_library
        self.scanned_books = scanned_books

    def export(self, file_path):
        with open(file_path, "w+") as ofp:
            ofp.write(f"{len(self.signed_libraries)}\n")
            for library in self.signed_libraries:
                # library_idx = int(library.split()[-1])
                books = self.scanned_books_per_library.get(library, [])
                ofp.write(f"{library} {len(books)}\n")
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


    def calculate_fitness_score(self, scores):
        score = 0
        for book in self.scanned_books:
            score += scores[book]
        self.fitness_score = score

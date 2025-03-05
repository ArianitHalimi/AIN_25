import random
from tqdm import tqdm
from .parser import Parser
import os

class Solution:
    signed_libraries = []
    unsigned_libraries = []
    scanned_books = {}  
    assigned_books = set()
    curr_time = 0
    output_file_name = None
    libraries_output_file_name = None

    def __init__(self, output_file_name):
        self.output_file_name = output_file_name
        self.libraries_output_file_name = self.output_file_name.replace(".txt", "")+"_libraries.txt"

    def solve(self, data):
        random.shuffle(data.libs)

        for library in tqdm(data.libs):
            if self.curr_time + library.signup_days >= data.num_days:  
                self.unsigned_libraries.append(f"Library {library.id}") 
                continue

            time_left = data.num_days - (self.curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                set(library.books) - self.assigned_books, key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:  
                self.signed_libraries.append(f"Library {library.id}")  
                self.scanned_books[library.id] = available_books
                self.assigned_books.update(available_books)
                self.curr_time += library.signup_days 

        all_books = set(range(data.num_books))  
        scanned_books_list = sorted(self.assigned_books)
        not_scanned_books_list = sorted(all_books - self.assigned_books)
        
        os.makedirs(os.path.dirname(self.libraries_output_file_name), exist_ok=True)

        with open(self.output_file_name, "w+") as lofp:
            lofp.write("Signed libraries: " + ", ".join(self.signed_libraries) + "\n")
            lofp.write("Unsigned libraries: " + ", ".join(self.unsigned_libraries) + "\n")
            lofp.write("\nScanned books per library:\n")
            for library_id, books in self.scanned_books.items():
                lofp.write(f"Library {library_id}: " + ", ".join(map(str, books)) + "\n")
            lofp.write("\nOverall scanned books: " + ", ".join(map(str, scanned_books_list)) + "\n")
            lofp.write("Not scanned books: " + ", ".join(map(str, not_scanned_books_list)) + "\n")
        
        print(f"Signed libraries: {len(self.signed_libraries)}")
        print(f"Unsigned libraries: {len(self.unsigned_libraries)}")
        print(f"Scanned books: {len(scanned_books_list)}")
        print(f"Not scanned books: {len(not_scanned_books_list)}")


        with open(f"{self.libraries_output_file_name}", "w+") as ofp:
            ofp.write(f"{len(self.signed_libraries)}\n")
            for library in self.signed_libraries:
                library_idx = int(library.split()[-1])
                books = self.scanned_books.get(library_idx, [])
                ofp.write(f"{library_idx} {len(books)}\n")
                ofp.write(" ".join(map(str, books)) + "\n")

        print(f"Processing complete! Output written to: {self.output_file_name}")
        print(f"Libraries summary saved to: {self.libraries_output_file_name}")
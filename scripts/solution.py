import random
from tqdm import tqdm
from models import Parser


parser = Parser('./input/input.txt')
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
scanned_books_list = sorted(assigned_books)
not_scanned_books_list = sorted(all_books - assigned_books)

libraries_output_file = 'libraries_output.txt'
with open(libraries_output_file, "w+") as lofp:
    lofp.write("Signed libraries: " + ", ".join(signed_libraries) + "\n")
    lofp.write("Unsigned libraries: " + ", ".join(unsigned_libraries) + "\n")
    lofp.write("\nScanned books per library:\n")
    for library_id, books in scanned_books.items():
        lofp.write(f"Library {library_id}: " + ", ".join(map(str, books)) + "\n")
    lofp.write("\nOverall scanned books: " + ", ".join(map(str, scanned_books_list)) + "\n")
    lofp.write("Not scanned books: " + ", ".join(map(str, not_scanned_books_list)) + "\n")
print(f"Signed libraries: {len(signed_libraries)}")
print(f"Unsigned libraries: {len(unsigned_libraries)}")
print(f"Scanned books: {len(scanned_books_list)}")
print(f"Not scanned books: {len(not_scanned_books_list)}")
output_file = 'result.txt'
with open(output_file, "w+") as ofp:
    ofp.write(f"{len(signed_libraries)}\n")
    for library in signed_libraries:
        library_idx = int(library.split()[-1])
        books = scanned_books.get(library_idx, [])
        ofp.write(f"{library_idx} {len(books)}\n")
        ofp.write(" ".join(map(str, books)) + "\n")
print(f"Processing complete! Output written to: {output_file}")
print(f"Libraries summary saved to: {libraries_output_file}")

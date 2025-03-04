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

import random
import sys

sys.path.append('../models')
from book import Book

class Library:
    id = 0
    num_books = 0
    signup_days = 0
    books_per_day = 0
    books = []
    _id_counter = 0

    def __init__(self, num_books, signup_days, books_per_day, books, book_scores):
        self.id = Library._id_counter
        Library._id_counter += 1
        self.num_books = num_books
        self.signup_days = signup_days
        self.books_per_day = books_per_day
        self.books = sorted([Book(x, book_scores[x]) for x in books], key=lambda x: x.score, reverse=True)

    def __repr__(self):
        return f"Library(id={self.id}, num_books={self.num_books}, signup_days={self.signup_days}, books_per_day={self.books_per_day}, books={self.books})"

def create_operator(libraries, schedule):
    """
    This operator adds a book from an unscheduled library to the schedule.

    Args:
        libraries: A list of Library objects.
        schedule: A list of Library objects representing the current schedule.
                  (In the simplest form, this could also be a list of tuples
                  if you also want to store the order of the books to be scanned)

    Returns:
        A tuple containing:
            - The modified schedule (list of Library objects).
            - A boolean indicating if the operation was successful.
    """

    unscheduled_libraries = [lib for lib in libraries if lib not in schedule]

    if not unscheduled_libraries:
        return schedule, False

    num_top_libraries = max(1, int(len(unscheduled_libraries) * 0.05))  # Ensure at least 1 library is selected
    top_libraries = sorted(unscheduled_libraries, key=lambda lib: sum(book.score for book in lib.books), reverse=True)[:num_top_libraries]

    print("\n--- Top 5% Selected Libraries ---")
    for lib in top_libraries:
        print(f"  {lib}")

    selected_library = random.choice(top_libraries)

    print("\n--- Single Library Selected Randomly ---")
    print(f"  {selected_library}")

    if not selected_library.books:
        return schedule, False
    selected_book = random.choice(selected_library.books)

    schedule.append(selected_library)

    return schedule, True

def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    B, L, D = map(int, lines[0].split())
    book_scores = list(map(int, lines[1].split()))

    libraries = []
    line_index = 2
    for _ in range(L):
        N, T, M = map(int, lines[line_index].split())
        book_ids = list(map(int, lines[line_index + 1].split()))
        libraries.append(Library(N, T, M, book_ids, book_scores))
        line_index += 2
    return libraries, B, D

libraries, total_books, days = load_data("../input/UPFIEK.txt")
schedule = []

schedule, success = create_operator(libraries, schedule)

print("\n--- Final Schedule ---")
for library in schedule:
    print(f"  {library}")

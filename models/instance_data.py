class InstanceData:
    num_books = 0
    num_libs = 0
    num_days = 0
    scores = []
    libs = []
    book_libs = []
    upper_bound = 0

    def __init__(self, num_books, num_libs, num_days, scores, libs):
        self.num_books = num_books
        self.num_libs = num_libs
        self.num_days = num_days
        self.scores = scores
        self.libs = libs
        self.book_libs = [[] for _ in range(num_books)]
        for i, lib in enumerate(libs):
            for book in lib.books:
                self.book_libs[book.id].append(i)

    def describe(self):
        print('There are', self.num_books, "books", self.num_libs, "libraries", "and", self.num_days, "days for scanning")
        print('The scores of the books are', ','.join(str(x) for x in self.scores), "(in order)")

        print()

        for i,l in enumerate(self.libs):
            print(f'Library {l.id} has {l.num_books} books, the signup process takes {l.signup_days} days, and the library can ship {l.books_per_day} books per day.')
            print(f'The books in library {l.id}  are: ' + ', '.join(f'book {x}' for x in l.books[:-1]) + f', and book {l.books[-1]}.')
            
        print()

        for i,l in enumerate(self.book_libs):
            print(f'Book {i} Exists in Libraries:', ' and '.join(str(x) for x in l))
            
    def calculate_upper_bound(self):
        """Calculates the sum of scores of all unique books across all libraries."""
        unique_books = set()
        
        # Collect all unique book IDs
        for lib in self.libs:
            for book in lib.books:
                unique_books.add(book.id)  

        # Sum up their scores
        upper_bound = sum(self.scores[book_id] for book_id in unique_books)
        return upper_bound


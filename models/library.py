class Library:
    id = 0
    num_books = 0
    signup_days = 0
    books_per_day = 0
    books = []
    
    _id_counter = 0

    def __init__(self, num_books, signup_days, books_per_day, books):
        self.id = Library._id_counter
        Library._id_counter += 1
        self.num_books = num_books
        self.signup_days = signup_days
        self.books_per_day = books_per_day
        self.books = books
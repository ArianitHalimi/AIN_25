from .library import Library
from .instance_data import InstanceData
import sys

class Parser:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def parse(self):
        try:
            with open(self.file_path, 'r') as file:
                try:
                    first_line = file.readline().strip()
                    if not first_line:
                        raise ValueError("File is empty or first line is missing")
                    
                    try:
                        parts = first_line.split(' ')
                        if len(parts) != 3:
                            raise ValueError(f"First line should contain exactly 3 integers, got {len(parts)}")
                        num_books, num_libs, num_days = map(int, parts)
                        if num_books < 0 or num_libs < 0 or num_days < 0:
                            raise ValueError(f"All values must be non-negative: books={num_books}, libraries={num_libs}, days={num_days}")
                    except ValueError as e:
                        if "invalid literal for int" in str(e):
                            raise ValueError("First line should contain integers only")
                        raise

                    scores_line = file.readline().strip()
                    if not scores_line:
                        raise ValueError("Book scores line is missing")
                    
                    try:
                        scores = list(map(int, scores_line.split()))
                        if len(scores) != num_books:
                            raise ValueError(f"Expected {num_books} book scores, got {len(scores)}")
                    except ValueError:
                        raise ValueError("Book scores must be integers")

                    libs = []
                    for i in range(num_libs):
                        lib_header = file.readline().strip()
                        if not lib_header:
                            raise ValueError(f"Library {i} header is missing")
                        
                        try:
                            parts = lib_header.split(' ')
                            if len(parts) != 3:
                                raise ValueError(f"Library {i} header should contain exactly 3 integers")
                            books_count, signup_days, books_per_day = map(int, parts)
                            if books_count < 0 or signup_days < 0 or books_per_day < 0:
                                raise ValueError(f"Library {i} values must be non-negative")
                        except ValueError:
                            raise ValueError(f"Library {i} header must contain integers only")
                        
                        books_line = file.readline().strip()
                        if not books_line:
                            raise ValueError(f"Book list for library {i} is missing")
                        
                        try:
                            books = list(map(int, books_line.split()))
                            if len(books) != books_count:
                                raise ValueError(f"Library {i} should have {books_count} books, got {len(books)}")
                            
                            if any(book_id < 0 or book_id >= num_books for book_id in books):
                                raise ValueError(f"Library {i} contains invalid book ID(s)")
                        except ValueError:
                            raise ValueError(f"Book IDs for library {i} must be integers")
                        
                        library = Library(books_count, signup_days, books_per_day, books, scores)
                        libs.append(library)

                    return InstanceData(num_books, num_libs, num_days, scores, libs)
                
                except ValueError as e:
                    print(f"Error parsing file: {str(e)}")
                    sys.exit(1)
                    # raise ValueError(f"Error parsing file: {str(e)}")
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            sys.exit(1)

            # raise FileNotFoundError(f"File not found: {self.file_path}")
        except PermissionError:
            print(f"Permission denied when accessing: {self.file_path}")
            sys.exit(1)

            # raise PermissionError(f"Permission denied when accessing: {self.file_path}")
        except Exception as e:
            print(f"Unexpected error when parsing file: {str(e)}")
            sys.exit(1)

            # raise Exception(f"Unexpected error when parsing file: {str(e)}")

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QSpacerItem, QSizePolicy
)

def read_input_file(input_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()
    B, L, D = map(int, lines[0].strip().split())
    book_scores = list(map(int, lines[1].strip().split()))
    libraries = []
    for i in range(2, len(lines), 2):
        if i + 1 >= len(lines):
            break
        N, T, M = map(int, lines[i].strip().split())
        books = set(map(int, lines[i + 1].strip().split()))
        libraries.append((N, T, M, books))
    return B, L, D, book_scores, libraries

def read_output_file(output_path):
    with open(output_path, 'r') as file:
        lines = file.readlines()
    num_libraries = int(lines[0].strip())
    solution = []
    index = 1
    for _ in range(num_libraries):
        if index >= len(lines):
            break
        lib_id, num_books = map(int, lines[index].strip().split())
        index += 1
        books = list(map(int, lines[index].strip().split()))
        index += 1
        solution.append((lib_id, num_books, books))
    return num_libraries, solution

def read_input_file(input_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()
    B, L, D = map(int, lines[0].strip().split())
    book_scores = list(map(int, lines[1].strip().split()))
    libraries = []
    for i in range(2, len(lines), 2):
        if i + 1 >= len(lines):
            break
        N, T, M = map(int, lines[i].strip().split())
        books = set(map(int, lines[i + 1].strip().split()))
        libraries.append((N, T, M, books))
    return B, L, D, book_scores, libraries

def read_output_file(output_path):
    with open(output_path, 'r') as file:
        lines = file.readlines()
    num_libraries = int(lines[0].strip())
    solution = []
    index = 1
    for _ in range(num_libraries):
        if index >= len(lines):
            break
        lib_id, num_books = map(int, lines[index].strip().split())
        index += 1
        books = list(map(int, lines[index].strip().split()))
        index += 1
        solution.append((lib_id, num_books, books))
    return num_libraries, solution

def validate_solution(input_path, output_path, isConsoleApplication = False):
    B, L, D, book_scores, libraries = read_input_file(input_path)
    num_libraries, solution = read_output_file(output_path)

    errors = []
    valid_libraries_info = []
    all_scanned_books = set()

    with open(output_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    expected_library_count = len(lines[1:]) // 2
    if num_libraries != expected_library_count:
        errors.append(f"Invalid solution: Declared {num_libraries} libraries, but output contains {expected_library_count} library entries.")

    if num_libraries > L:
        errors.append(f"Invalid solution: Output references {num_libraries} libraries, but only {L} exist.")

    assigned_books = set()
    used_libraries = set()
    total_days_used = 0
    total_score = 0
    libraries_used = 0

    for i, entry in enumerate(solution):
        if not entry or len(entry) < 2:
            errors.append(f"Missing or malformed entry for a library at line {i * 2 + 1}.")
            continue

        lib_id = entry[0]
        num_books = entry[1]
        books = entry[2] if len(entry) > 2 else []

        if lib_id >= L:
            errors.append(f"Library {lib_id} does not exist.")
            continue
        
        N, T, M, library_books = libraries[lib_id]

        if total_days_used + T >= D:
            errors.append(f"Library {lib_id} takes too long to sign up ({T} days), leaving no time for scanning.")
            continue  

        total_days_used += T  

        if num_books != len(books):
            errors.append(f"Library {lib_id}: Declared {num_books} books, but actually listed {len(books)} books in output file.")
        
        invalid_books = [b for b in books if b not in library_books]
        if invalid_books:
            errors.append(f"Library {lib_id} contains invalid books: {invalid_books}.")
        
        if lib_id in used_libraries:
            errors.append(f"Library {lib_id} is listed multiple times in the solution.")
        
        used_libraries.add(lib_id)
        
        # Ensure books are scanned only once globally
        unique_books = [b for b in books if b not in all_scanned_books]
        all_scanned_books.update(unique_books)
        
        assigned_books.update(unique_books)
        libraries_used += 1

        remaining_days = D - total_days_used
        if remaining_days <= 0:
            errors.append(f"Library {lib_id} has no time left for scanning books after sign-up.")
            continue

        max_possible_books = min(remaining_days * M, len(library_books))
        if len(unique_books) > max_possible_books:
            errors.append(f"Library {lib_id} attempts to scan {len(unique_books)} books, exceeding the limit of {max_possible_books}.")

        total_score += sum(book_scores[b] for b in unique_books if 0 <= b < len(book_scores))

        scanned_books = [(b, book_scores[b] if 0 <= b < len(book_scores) else 0) for b in unique_books]
        valid_libraries_info.append(
            f"Library {lib_id}:"
            f"Scanned books: {', '.join([f'book {b} (score {s})' for b, s in scanned_books])}"
        )

    all_available_books = set(range(B))
    not_scanned_books = all_available_books - all_scanned_books
    not_scanned_books_sorted = sorted(not_scanned_books, key=lambda b: book_scores[b] if 0 <= b < len(book_scores) else 0, reverse=True)
    not_scanned_books_str = ", ".join([f"book {b} (score {book_scores[b] if 0 <= b < len(book_scores) else 0})" for b in not_scanned_books_sorted])

    # Fitness Score Calculation
    total_possible_score = sum(book_scores)  # Max possible score if all books were scanned

    score_efficiency = total_score / total_possible_score if total_possible_score > 0 else 0
    library_utilization = libraries_used / L if L > 0 else 0
    book_scanning_efficiency = len(all_scanned_books) / B if B > 0 else 0

    fitness_score = (0.5 * score_efficiency) + (0.3 * library_utilization) + (0.2 * book_scanning_efficiency)

    if errors:
        return "\n".join(errors)
    
    result = (
    f"Solution is valid!\n"
    f"Total score: {total_score}\n"
    f"Fitness score: {fitness_score:.4f}\n\n"
    f"Signed up libraries: {libraries_used}/{L} ({(libraries_used / L) * 100:.2f}%)\n"
    f"Unigned up libraries: {L-libraries_used}/{L} ({((L-libraries_used) / L) * 100:.2f}%)\n"
    f"Scanned books: {len(all_scanned_books)}/{B} ({(len(all_scanned_books) / B) * 100:.2f}%)\n"
    f"Unscanned books: {len(not_scanned_books)}/{B} ({(len(not_scanned_books) / B) * 100:.2f}%)\n"
    f"Used days: {total_days_used}/{D} ({total_days_used / B * 100:.2f}%)\n\n"
    )

    if isConsoleApplication:
        result += "For more info run script without providing any file path (UI).\n\n"
    else:
        result += "\n".join(valid_libraries_info)
    
    return result

class ValidatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Hash Code 2020 Validator')
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()
        
        self.input_label = QLabel("Input file: None")
        layout.addWidget(self.input_label)
        self.input_button = QPushButton("Browse Input File")
        self.input_button.clicked.connect(self.browse_input)
        layout.addWidget(self.input_button)
        
        self.output_label = QLabel("Output file: None")
        layout.addWidget(self.output_label)
        self.output_button = QPushButton("Browse Output File")
        self.output_button.clicked.connect(self.browse_output)
        layout.addWidget(self.output_button)
        
        spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        layout.addItem(spacer)

        self.validate_button = QPushButton("Validate Solution")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        
        self.setLayout(layout)
        self.input_path = None
        self.output_path = None
    
    def browse_input(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Input File", "", "Text Files (*.txt)")
        if file_name:
            self.input_path = file_name
            self.input_label.setText(f"Input file: {os.path.basename(file_name)}")
    
    def browse_output(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Output File", "", "Text Files (*.txt)")
        if file_name:
            self.output_path = file_name
            self.output_label.setText(f"Output file: {os.path.basename(file_name)}")
    
    def validate(self):
        if not self.input_path or not self.output_path:
            self.result_text.setText("Please select both input and output files.")
            return
        result = validate_solution(self.input_path, self.output_path)
        self.result_text.setText(result)

def main():
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        result = validate_solution(input_path, output_path, True)
        print(result)
    else:
        app = QApplication(sys.argv)
        validator = ValidatorApp()
        validator.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    main()

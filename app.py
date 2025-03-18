from models import Parser
from models import Solver

import os

solver = Solver()

directory = os.listdir('input')

print("---------- RANDOM SEARCH ----------")
for file in directory:
    if file.endswith('.txt'):
        parser = Parser(f'./input/{file}')
        data = parser.parse()
        best_solution_file = solver.random_search(data)
        print(best_solution_file[0], file)

# solution.export('./output/output.txt')

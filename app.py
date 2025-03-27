from models import Parser
from models import Solver

import os

solver = Solver()

directory = os.listdir('input')

# print("---------- RANDOM SEARCH ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         best_solution_file = solver.random_search(data)
#         print(best_solution_file[0], file)

print("---------- HILL CLIMBING SWAP SIGNED ----------")
for file in directory:
    if file.endswith('.txt'):
        parser = Parser(f'./input/{file}')
        data = parser.parse()
        hill_climbing_signed = solver.hill_climbing_signed(data, file)
        print(hill_climbing_signed[0], file)

print("---------- HILL CLIMBING SIGNED & UNSIGNED SWAP ----------")
for file in directory:
    if file.endswith('.txt'):
        print(f'Computing ./input/{file}')
        parser = Parser(f'./input/{file}')
        data = parser.parse()
        solution = solver.hill_climbing_signed_unsigned(data)
        solution.export('./output/output.txt')
        print(f"{solution.fitness_score:,}", file)

# solution.export('./output/output.txt')

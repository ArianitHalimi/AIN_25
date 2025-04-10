from models import Parser
from models import Solver


import os
import time
# import tkinter as tk
# from tkinter import messagebox

solver = Solver()

directory = os.listdir('input')

# print("---------- RANDOM SEARCH ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         best_solution_file = solver.random_search(data)
#         print(best_solution_file[0], file)

# print("---------- HILL CLIMBING SWAP SIGNED ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         hill_climbing_signed = solver.hill_climbing_signed(data, file)
#         print(hill_climbing_signed[0], file)

# print("---------- HILL CLIMBING SIGNED & UNSIGNED SWAP ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         print(f'Computing ./input/{file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         solution = solver.hill_climbing_signed_unsigned(data)
#         # solution.export('./output/output.txt')
#         print(f"{solution.fitness_score:,}", file)

# solution.export('./output/output.txt')

# print("---------- HILL CLIMBING SWAP LAST BOOK ----------")
# for file in directory:
#     if file.endswith('f_libraries_of_the_world.txt'):
#         print(f'Computing ./input/{file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         solution = solver.hill_climbing_swap_last_book(data)[1]
#         # solution.export(f'./output/{file}')
#         print(f"{solution.fitness_score:,}", file)
#
# solution.export('./output/output.txt')

# files = ['f_libraries_of_the_world.txt','d_tough_choices.txt']
# for file in directory:
#     if file.endswith('.txt'):
#         print(f'Computing ./input/{file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         solution = solver.hill_climbing_combined(data)
#         print(solution)


# results = []
# for file in directory:
#     if file.endswith('.txt'):
#         print(f'Computing ./input/{file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         solution = solver.hill_climbing_combined(data)
#         # solution.export('./output/output.txt')


# # print("Best Solution:")
# # results.sort(reverse=True)
# # for score, file in results:
# #     print(f"{score:,}", file)

# # Create a hidden root window
# root = tk.Tk()
# root.withdraw()

# # results.append((2222, 'test'))  # Placeholder for the best solution
# # #3 more placeholders
# # results.append((3333, 'test2'))  # Placeholder for the best solution
# # results.append((4444, 'test3'))  # Placeholder for the best solution

# # Display results all in one message box
# message = "Best Solutions:\n"
# for score, file in results:
#     message += f"{file}: {score:,}\n"
# messagebox.showinfo("Best Solutions", message)

# # Destroy the root window when done
# root.destroy()

# print("results", results)


# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()

#         # Calculate upper bound
#         upper_bound = data.calculate_upper_bound()
#         print(f"Upper Bound (Sum of Scores of Unique Books) for {file}: {upper_bound}")


# print("---------- Hill-Climbing Swap Same Books with Crossover----------")
# timeout_duration = 30 * 60

# for file in directory:

#     if file.endswith('.txt'):
#         start_time = time.time()
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         solver = Solver()
#         initial_solution = solver.generate_initial_solution(data)
#         optimized_solution = solver.hill_climbing_with_crossover(initial_solution, data)
#         # optimized_solution.export('./output/output.txt')
#         end_time = time.time()
#         elapsed_time = end_time - start_time

#         print(f"Best Fitness Score for {file}: {optimized_solution.fitness_score}")
#         print(f"Time taken for {file}: {elapsed_time:.2f} seconds")

#         if elapsed_time > timeout_duration:
#             print(f"Timeout reached for {file}, stopping processing.")
#             break  # Stop processing further files if timeout is exceeded

# print("---------- ITERATED LOCAL SEARCH WITH RANDOM RESTARTS ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         print(f'Computing ./input/{file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         result = solver.iterated_local_search(data, time_limit=300, max_iterations=1000)
#         print(f"Final score for {file}: {result[0]:,}")
#         output_dir = 'output/ils_random_restarts'
#         os.makedirs(output_dir, exist_ok=True)
#         output_file = os.path.join(output_dir, file)
#         result[1].export(output_file)
#         print("----------------------")


# Hill climbing with inserts
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         score, solution = solver.hill_climbing_with_random_restarts(data, total_time_ms=1000)
        
#         solution.export(f'./output/{file}')
#         print(f'Final score: {score:,}')
#         print(f'Solution exported to ./output/{file}')

# print("---------- STEEPEST ASCENT HILL CLIMBING ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         print(parser)
#         data = parser.parse()
        
#         score, solution = solver.steepest_ascent_hill_climbing(data, n=5, total_time_ms=1000)
#         solution.export(f'./output/{file}')
#         print(f'Final score: {score:,}')

# print("---------- BEST OF TWO BETWEEN STEEPEST ASCENT AND RANDOM RESTART ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         score, solution = solver.best_of_steepest_ascent_and_random_restart(data, total_time_ms=1000)
        
#         solution.export(f'./output/best-of-two/{file}')
#         print(f'Final score: {score:,}')
#         print(f'Solution exported to ./output/best-of-two/{file}')


# print("---------- MONTE CARLO SEARCH ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         print(f'Computing ./input/{file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         score, solution = solver.monte_carlo_search(data, num_iterations=1000, time_limit=60)
#         solution.export(f'./output/monte_carlo_{file}')
#         print(f'Final score: {score:,}')
#         print(f'Solution exported to ./output/monte_carlo_{file}')

# Hill climbing with inserts
# for file in directory:
#     if file.endswith('.txt'):
#         print(f'Processing file: {file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()

#         # Call the hill_climbing_insert_library function
#         score, solution = solver.hill_climbing_insert_library(data, iterations=1000)

#         # Export the solution
#         solution.export(f'./output/{file}')
#         print(f'Final score for {file}: {score:,}')
#         print(f'Solution exported to ./output/{file}')

print("---------- GUIDED LOCAL SEARCH ----------")
for file in directory:
    if file.endswith('.txt'):
        print(f'Processing file: {file}')
        parser = Parser(f'./input/{file}')
        data = parser.parse()

        # Call the guided local search function
        solution = solver.guided_local_search(data, max_time=300, max_iterations=1000)

        # Export the solution
        solution.export(f'./output/gls_{file}')
        print(f'Final score for {file}: {solution.fitness_score:,}')
        print(f'Solution exported to ./output/gls_{file}')



# print("---------- STEEPEST ASCENT HILL CLIMBING ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         print(parser)
#         data = parser.parse()
        
#         score, solution = solver.steepest_ascent_hill_climbing(data, n=5, total_time_ms=1000)
#         solution.export(f'./output/{file}')
#         print(f'Final score: {score:,}')

# print("---------- BEST OF TWO BETWEEN STEEPEST ASCENT AND RANDOM RESTART ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         score, solution = solver.best_of_steepest_ascent_and_random_restart(data, total_time_ms=1000)
        
#         solution.export(f'./output/best-of-two/{file}')
#         print(f'Final score: {score:,}')
#         print(f'Solution exported to ./output/best-of-two/{file}')
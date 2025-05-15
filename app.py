import csv
from models import Parser
from models import Solver


import os, sys
import time
import multiprocessing
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

# files = ['a_example.txt','b_read_on.txt']
# for file in files:
#     if file.endswith('.txt'):
#         print(f'Computing ./input/{file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         solution = solver.hill_climbing_combined(data)
#         print(solution[0])

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

# print("---------- Tabu Search ----------")

# for file in directory:

#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         solver = Solver()
#         initial_solution = solver.generate_initial_solution_grasp(data)
#         optimized_solution = solver.tabu_search(initial_solution, data, tabu_max_len=10, n=5, max_iterations=100)

#         # optimized_solution.export('./output/output.txt')

#         print(f"Best Fitness Score for {file}: {optimized_solution.fitness_score}")


# print("---------- Feature-based Tabu Search ----------")

# parser = Parser(f'./input/e_so_many_books.txt')
# data = parser.parse()
# solver = Solver()
# optimized_solution = solver.hill_climbing_with_random_restarts_basic(data, 30000)

# initial_solution = solver.generate_initial_solution_grasp(data)
# optimized_solution = solver.feature_based_tabu_search(initial_solution, data, tabu_max_len=10, n=5, max_iterations=100)

# optimized_solution.export('./output/output.txt')

# print(f"Best Fitness Score for: {optimized_solution[1]}")


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

# print("---------- GUIDED LOCAL SEARCH ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         print(f'Processing file: {file}')
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()

#         # Call the guided local search function
#         solution = solver.guided_local_search(data, max_time=300, max_iterations=1000)

#         # Export the solution
#         solution.export(f'./output/gls_{file}')
#         print(f'Final score for {file}: {solution.fitness_score:,}')
#         print(f'Solution exported to ./output/gls_{file}')



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

# print("---------- Simulated Annealing With Cutoff ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         parser = Parser(f'./input/{file}')
#         data = parser.parse()
#         score, solution = solver.simulated_annealing_with_cutoff(data, total_time_ms=1000)
#
#         solution.export(f'./output/{file}')
#         print(f'Final score: {score:,}')
#         print(f'Solution exported to ./output/{file}')

# input_folder = './input'
# output_folder = './output'
# os.makedirs(output_folder, exist_ok=True)

# instance_files = [
#     'UPFIEK.txt',
#     'a_example.txt',
#     'b_read_on.txt',
#     'c_incunabula.txt',
#     'd_tough_choices.txt',
#     'e_so_many_books.txt',
#     'f_libraries_of_the_world.txt',
#     'Toy instance.txt',
#     'B5000_L90_D21.txt',
#     'B50000_L400_D28.txt',
#     'B100000_L600_D28.txt',
#     'B90000_L850_D21.txt',
#     'B95000_L2000_D28.txt',
#     'switch_book_instance.txt'
# ]

# print("---------- VARIABLE NEIGHBORHOOD SEARCH ----------")

# for filename in os.listdir(input_folder):
#     if filename.endswith('.txt'):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, f'vns_{filename}')

#         try:
#             print(f"Parsing {filename}...")
#             parser = Parser(input_path)
#             data = parser.parse()

#             print(f"Running VNS on {filename}...")

#             # Run VNS algorithm
#             score, solution = solver.variable_neighborhood_search(data, time_limit_ms=10000)

#             # Export the solution
#             solution.export(output_path)

#             print(f'Final VNS score for {filename}: {score:,}')
#             print(f'Solution exported to: {output_path}')
#             print('-' * 50)

#         except Exception as e:
#             print(f" Error processing {filename}: {e}")

# print("---------- GREAT DELUGE ALGORITHM ----------")
# for file in directory:
#     if file.endswith('.txt'):
#         if file in ["c_incunabula.txt"]:
#             parser = Parser(f'./input/{file}')
#             print(parser)
#             data = parser.parse()

#             start_time = time.time()
#             score, solution = solver.enhanced_great_deluge_algorithm(data)
#             end_time = time.time()
#             elapsed_time = end_time - start_time

#             solution.export(f'./output/gda-simple/{file}')
#             print(f'Final score: {score:,}')
#             print(f'Time taken: {elapsed_time:.2f} seconds')

#             with open('./output/gda-simple/notes-simple.txt', 'a') as notes:
#                 notes.write(f'From: {file}\n')
#                 notes.write(f'Final score: {score:,}\n')
#                 notes.write(f'Time taken: {elapsed_time:.2f} seconds\n\n')

# print("---------- PARALLELED GREAT DELUGE ALGORITHM ----------")
# with open('./output/parallel-gda/optimization_log.csv', 'w') as log:
#     log.write("filename,score,time_seconds,phases_executed,solution_diversity\n")
    
# for file in directory:
#     if file in ["a_example.txt"]:
#         parser = Parser(f'./input/{file}')
#         print(parser)
#         data = parser.parse()

#         start_time = time.time()
#         # Get runner instance along with results
#         gda_runner, score, solution = solver.run_cpu_optimized_gda(data, max_time=300)
#         end_time = time.time()
#         elapsed_time = end_time - start_time

#         # Generate performance analysis
#         gda_runner.analyze_performance()
        
#         # Decision for extended run
#         print("should continue value: ", gda_runner.should_continue(score))
#         if gda_runner.should_continue(score):
#             print("\nSignificant potential detected - extending run...")
#             ext_score, ext_solution = gda_runner.run_iterative_phases(600)  # 10 more minutes
#             if ext_score > score:
#                 score, solution = ext_score, ext_solution
#                 elapsed_time = time.time() - start_time
#                 print(f"Improved score after extension: {score}")
#             else:
#                 print("Extension didn't improve results")

#         solution.export(f'./output/parallel-gda/{file}')
        
#         metrics = {
#             'file': file,
#             'score': score,
#             'time': elapsed_time,
#             'phases': len(gda_runner.phase_history),
#             'diversity': gda_runner.phase_history[-1][3] if gda_runner.phase_history else 0.0
#         }
        
#         # Console output
#         print(f"\nFINAL RESULTS FOR {file}")
#         print(f"Best score: {metrics['score']:,}")
#         print(f"Total time: {metrics['time']:.2f}s")
#         print(f"Phases executed: {metrics['phases']}")
#         print(f"Solution diversity: {metrics['diversity']:.2f}")
        
#         # CSV logging
#         with open('./output/parallel-gda/optimization_log.csv', 'a') as log:
#             writer = csv.writer(log)
#             writer.writerow([
#                 metrics['file'],
#                 metrics['score'],
#                 f"{metrics['time']:.2f}",
#                 metrics['phases'],
#                 f"{metrics['diversity']:.2f}"
#             ])
            
#         print(f"\nCompleted processing {file}\n")

# if __name__ == '__main__':
#     print("---------- HYBRID PARALLEL EVOLUTIONARY SEARCH ----------")
#     for file in directory:
#         if file.endswith('.txt'):
#             print(f'Computing ./input/{file}')
#             parser = Parser(f'./input/{file}')
#             data = parser.parse()
            
#             score, solution = solver.hybrid_parallel_evolutionary_search(
#                 data, 
#                 num_iterations=1000, 
#                 time_limit=60
#             )
#             solution.export(f'./output/hybrid_evolutionary_{file}')
#             print(f'Final score: {score:,}')
#             print(f'Solution exported to ./output/hybrid_evolutionary_{file}')
# def run_parallel_sa():

#     print("---------- SIMULATED ANNEALING WITH MULTIPLE TEMPERATURE FUNCTIONS (PARALLEL) ----------")
#     for file in directory:
#         if file.endswith('.txt'):
#             print(f'Computing ./input/{file}')
#             parser = Parser(f'./input/{file}')
#             data = parser.parse()
#             score, solution = solver.simulated_annealing_hybrid_parallel(data, max_iterations=1000)
#             print(f'Best score from SA (parallel) for {file}: {score:,}')
#             output_file = f'./output/sa_hybrid_parallel_{file}'
#             solution.export(output_file)
#             print(f"Processing complete! Output written to: {output_file}")
           
# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     run_parallel_sa()


# if __name__ == '__main__':
#     _directory = './Large-scale test set (50 instances)'
#     ub_folder = './output/upper_bounds'
#     os.makedirs(ub_folder, exist_ok=True)

#     if not os.path.isdir(_directory):
#         print(f"Informatë: Folderi '{_directory}' nuk është gjetur.")
#         sys.exit(1)

#     instances = sorted(
#         f for f in os.listdir(_directory)
#         if f.endswith('.txt') or f.endswith('.in')
#     )

#     if not instances:
#         print(f"Informatë: Nuk u gjetën fajlla .txt ose .in në '{_directory}'.")
#         sys.exit(1)

#     report_path = os.path.join(ub_folder, 'upper_bounds.txt')
#     with open(report_path, 'w') as report:
#         header = f"{'Instance':<40} {'Time-Aware':>12} {'Greedy':>12} {'Library-Only':>14}"
#         separator = '-' * 80

#         report.write(header + "\n")
#         report.write(separator + "\n")
#         print(header)
#         print(separator)

#         for fname in instances:
#             full_path = os.path.join(_directory, fname)
#             time_aware = solver.time_aware_score(full_path)
#             greedy = solver.greedy_score(full_path)
#             library_only = solver.library_only_score(full_path)

#             line = f"{fname:<40} {time_aware:12} {greedy:12} {library_only:14}"
#             report.write(line + "\n")
#             print(line)

#     print(f"\nRaporti është ruajtur në: {report_path}")

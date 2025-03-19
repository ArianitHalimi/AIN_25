from models import Parser
from models import Solver

parser = Parser('./input/input.txt')
data = parser.parse()

# data.describe()

solver = Solver()
# solution = solver.generateInitialSolution(data)
solution = solver.hill_climbing(data)
solution.export('./output/output.txt')

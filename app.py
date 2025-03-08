from models import Parser
from models import Solver

parser = Parser('./input/d_tough_choices.txt')
data = parser.parse()

# data.describe()

solver = Solver()
solution = solver.solve(data)
# solution.export('./output/output.txt')

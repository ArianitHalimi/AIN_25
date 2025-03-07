from models import Parser
from models import Solution

parser = Parser('./input/input.txt')
data = parser.parse()

data.describe()

solution = Solution('./output/output.txt')
solution.solve(data)

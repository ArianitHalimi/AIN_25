from models import Parser

parser = Parser('./input/input.txt')
data = parser.parse()

data.describe()

class Book:
    id = 0
    score = 0

    def __init__(self, id, score):
        self.id = id    
        self.score = score

    def __repr__(self):
        return f"Book({self.id}, {self.score})"

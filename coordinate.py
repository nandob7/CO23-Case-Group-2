import math


class Coordinate:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def distance(self, other):
        distance = math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
        return math.floor(distance)

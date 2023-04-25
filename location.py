import math


class Location:
    def __init__(self, i: int, x: int, y: int):
        self.lid = i
        self.x = x
        self.y = y
        self.stored_tools = {}

    def distance(self, other):
        distance = math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
        return math.floor(distance)

    def add_tool(self, tid, n):
        self.stored_tools.update({tid: self.stored_tools.get(tid) + n})

    def remove_tool(self, tid, n):
        self.stored_tools.update({tid: self.stored_tools.get(tid) - n})

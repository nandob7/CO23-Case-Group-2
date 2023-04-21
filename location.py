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

    def add_tool(self, tool):
        if tool in self.stored_tools.keys():
            self.stored_tools.update({tool: self.stored_tools.get(tool)+1})
        else:
            self.stored_tools.update({tool: 1})

    def remove_tool(self, tool):
        stored_no = self.stored_tools.get(tool)
        if stored_no == 1:
            self.stored_tools.pop(tool)
        else:
            self.stored_tools.update({tool: stored_no - 1})

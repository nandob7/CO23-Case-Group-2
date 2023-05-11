class Tool:
    def __init__(self, tid, size, max_no, cost, days=1):
        self.tid = tid
        self.size = size
        self.used = False
        self.in_use = [0 for d in range(days)]
        self.lid = 0
        self.max_no = max_no
        self.cost = cost

    def copy(self, days):
        copy = Tool(self.tid, self.size, self.max_no, self.cost, days)
        return copy

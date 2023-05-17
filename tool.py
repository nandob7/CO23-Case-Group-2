class Tool:
    def __init__(self, tid, size, max_no, cost, days):
        self.tid = tid
        self.id = 0
        self.size = size
        self.in_use = [False] * days
        self.max_no = max_no
        self.cost = cost


class Tool:
    def __init__(self, tid, size, max_no, cost):
        self.tid = tid
        self.size = size
        self.in_use = 0
        self.lid = 0
        self.max_no = max_no
        self.cost = cost

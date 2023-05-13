class Tool:
    def __init__(self, tid, size, max_no, cost):
        self.tid = tid
        self.id = 0
        self.size = size
        self.available_from = 0
        self.old_available = 0
        self.max_no = max_no
        self.cost = cost


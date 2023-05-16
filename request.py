class Request:
    def __init__(self, rid, lid, first, last, stay, tid, units):
        self.rid = rid
        self.lid = lid
        self.first = first
        self.last = last
        self.stay = stay
        self.tid = tid
        self.units = units
        self.start = None
        self.complete = None
        self.pickup = None
        self.tools = []
        self.vid = [0, 0]

    def deliver(self, day):
        self.pickup = day + self.stay

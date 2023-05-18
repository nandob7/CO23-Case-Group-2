class Request:
    def __init__(self, rid, lid, first, last, stay, tid, units):
        self.rid = rid
        self.lid = lid
        self.first = first
        self.last = last
        self.stay = stay
        self.tid = tid
        self.units = units
        self.pickup = None

    # Calculates the pickup day when the delivery is scheduled
    def deliver(self, day):
        self.pickup = day + self.stay

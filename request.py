class Request:

    pickup = None

    def __init__(self, rid, lid, first, last, stay, tid, no_tools):
        self.rid = rid
        self.lid = lid
        self.first = first
        self.last = last
        self.stay = stay
        self.tid = tid
        self.no_tools = no_tools

    def deliver(self, day):
        self.pickup = day + self.stay

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
        self.pickup = None
        self.priority = 0
        self.vid = [0, 0]
        self.twindow = list(range(self.first, self.last + 1))

    def deliver(self, day):
        self.pickup = day + self.stay

    def priority_calc(self, max_twindows, max_stay, tools, distances, max_dist):
        # if day == self.pickup:
        #     self.priority = 100
        # else:
        # twindow_prio = (self.last - day) / self.twindow
        stay_prio = (self.stay / max_stay[self.tid - 1])
        no_tool_prio = self.no_tools / tools[self.tid - 1].max_no
        distance_prio = distances[0, self.lid] / max_dist
        self.priority = -5 * (len(self.twindow) / max_twindows[self.tid - 1])\
                        + 3 * stay_prio + 2 * no_tool_prio + 1 * distance_prio

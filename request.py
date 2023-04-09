class Request:

    pickup = None

    def __init__(self, rid, loc_id, first, last, stay, tool_id, no_tools):
        self.rid = rid
        self.loc_id = loc_id
        self.first = first
        self.last = last
        self.stay = stay
        self.tool_id = tool_id
        self.no_tools = no_tools

    def deliver(self, day):
        self.pickup = day + self.stay

class Day:
    def __init__(self):
        self.requests = []
        self.routes = []
        self.depot_tools = dict()
        self.mileage = 0

    def calc_mileage(self):
        for r in self.routes:
            self.mileage += r.mileage

        return self.mileage

    def schedule(self, request, route):
        self.requests.append(request)
        self.routes.append(route)
        self.depot_tools[request.tid] -= request.no_tools \
            if request.pickup is None else - request.no_tools

    def unschedule(self, request):
        self.requests.pop()
        self.routes.pop()
        self.depot_tools[request.tid] += request.no_tools \
            if request.pickup is None else -request.no_tools


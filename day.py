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

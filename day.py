import solver


class Day:
    def __init__(self):
        self.requests = []
        self.routes = []
        self.cost = 0

    def calc_costs(self):
        distance_cost = 0
        for r in self.routes:
            distance_cost += r.calculate_route_cost()
        return len(self.routes) * solver.VEHICLE_DAY_COST + distance_cost

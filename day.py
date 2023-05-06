class Day:
    def __init__(self):
        self.requests = []
        self.routes = []
        self.depot_tools = list()
        self.cost = 0

    def calc_costs(self, v_d_cost):
        distance_cost = 0
        for r in self.routes:
            distance_cost += r.calculate_route_cost()
        self.cost = len(self.routes) * v_d_cost + distance_cost

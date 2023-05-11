class Vehicle:
    # Initialize an empty vehicle at the depot
    def __init__(self, vid, days):
        self.vid = vid
        self.load = 0
        self.active = [0 for d in range(days)]
        self.mileage = 0
        self.active_days = 0
        self.cost = 0

    def assign_route(self, route):
        self.active[route.day - 1] = 1
        self.active_days += 1
        self.mileage += route.mileage
        route.vid = self.vid

    def calculate_vehicle_cost(self, v_d_cost, v_cost):
        return self.active_days * v_d_cost + v_cost

    def reset_load(self):
        self.load = 0

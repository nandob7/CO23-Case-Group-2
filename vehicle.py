class Vehicle:
    # Initialize an empty vehicle at the depot
    def __init__(self, vid):
        self.vid = vid
        self.load = 0
        self.mileage = 0
        self.active_days = 0
        self.cost = 0

    def assign_route(self, route):
        self.active_days += 1
        self.mileage += route.mileage

    def calculate_vehicle_cost(self, v_d_cost, v_cost):
        return self.active_days * v_d_cost + v_cost

    def reset_load(self):
        self.load = 0

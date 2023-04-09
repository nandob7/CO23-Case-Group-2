from solver import VEHICLE_COST, VEHICLE_DAY_COST, DISTANCE_COST


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

    def calculate_vehicle_cost(self):
        return self.active_days * VEHICLE_DAY_COST + VEHICLE_COST # + mileage * DISTANCE_COST

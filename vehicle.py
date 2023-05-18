class Vehicle:
    # Initialize an empty vehicle at the depot
    def __init__(self, vid, days):
        self.vid = vid
        self.load = 0
        self.active = [0 for d in range(days)]
        self.mileage = 0
        self.cost = 0

    # Assigns a route to a vehicle, adds mileage, and vid id to the route
    def assign_route(self, route):
        self.active[route.day - 1] = 1
        self.mileage += route.mileage
        route.vid = self.vid

    def reset_load(self):
        self.load = 0

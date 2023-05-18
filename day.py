class Day:
    def __init__(self):
        self.routes = []
        self.mileage = 0

    # Calculates the mileage of a day
    def calc_mileage(self):
        for r in self.routes:
            self.mileage += r.mileage

        return self.mileage

    # Adds a route to the day's list of routes for final output purposes
    def schedule(self, route):
        self.routes.append(route)

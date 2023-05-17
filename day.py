class Day:
    def __init__(self):
        self.routes = []
        self.mileage = 0

    def calc_mileage(self):
        for r in self.routes:
            self.mileage += r.mileage

        return self.mileage

    def schedule(self, route):
        self.routes.append(route)

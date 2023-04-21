# Define a class to represent a route that contains a list of visited locations and the distance travelled
from solver import MAX_TRIP_DISTANCE, DISTANCE_COST


class Route:
    # Initialize an empty route with a list to store the visited locations and mileage
    def __init__(self):
        self.day = 0
        self.visited = []
        self.mileage = 0

    # Add a visited location and extra mileage to the route
    def add_visit(self, lid, distances):
        self.mileage += distances[lid, self.visited[-1]]
        self.visited.append(lid)

    def calculate_route_cost(self):
        return self.mileage * DISTANCE_COST

    def possible_addition(self, location, distances):
        return self.mileage + distances[self.visited[-1], location.lid] + distances[location.lid, 0]\
            <= MAX_TRIP_DISTANCE

# Define a class to represent a route that contains a list of visited locations and the distance travelled

class Route:
    # Initialize an empty route with a list to store the visited locations and mileage
    def __init__(self, day):
        self.day = day
        self.visited = [0]
        self.mileage = 0
        self.vid = 0

    # Add a visited location and extra mileage to the route
    def add_visit(self, request, distances, is_pickup, requests):
        prev = 0
        if self.visited[-1] != 0:
            prev = requests[abs(self.visited[-1]) - 1].lid
        self.mileage += distances[request.lid, prev]

        if not is_pickup:
            self.visited.append(request.rid)
        else:
            self.visited.append(-request.rid)

    # To be used at the end of a route, adds the trip back to the depot
    def back_to_depot(self, reqs, distances):
        self.mileage += distances[reqs[abs(self.visited[-1]) - 1].lid, 0]
        self.visited.append(0)

    # Checks the possibility of adding a certain request to a route considering the added distance vs max distance
    def possible_addition(self, request, distances, max_dist, reqs):
        return self.mileage + distances[reqs[abs(self.visited[-1]) - 1].lid, request.lid] + distances[request.lid, 0] \
            <= max_dist

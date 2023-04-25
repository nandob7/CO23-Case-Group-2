# Define a class to represent a route that contains a list of visited locations and the distance travelled

class Route:
    # Initialize an empty route with a list to store the visited locations and mileage
    def __init__(self, day):
        self.day = day
        self.visited = [0]
        self.mileage = 0
        self.vid = 0

    # Add a visited location and extra mileage to the route
    def add_visit(self, request, distances, tools):
        self.mileage += distances[request.lid, self.visited[-1]]
        if request.first == self.day:
            self.visited.append(request.rid)
            for i in range(request.no_tools):
                for t in tools:
                    if t.tid == request.tid and not t.in_use:
                        t.in_use = True
                        t.used = True
                        t.lid = request.lid
        else:
            self.visited.append(-request.rid)
            for i in range(request.no_tools):
                for t in tools:
                    if t.lid == request.lid and t.in_use:
                        t.in_use = False
                        t.lid = 0

    def back_to_depot(self, reqs, distances):
        self.mileage += distances[reqs[abs(self.visited[-1])-1].lid, 0]
        self.visited.append(0)

    def calculate_route_cost(self, d_cost):
        return self.mileage * d_cost

    def possible_addition(self, location, distances, max_dist):
        return self.mileage + distances[self.visited[-1], location.lid] + distances[location.lid, 0] \
            <= max_dist

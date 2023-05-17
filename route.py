# Define a class to represent a route that contains a list of visited locations and the distance travelled

class Route:
    # Initialize an empty route with a list to store the visited locations and mileage
    def __init__(self, day):
        self.day = day
        self.visited = [0]
        self.mileage = 0
        self.vid = 0

    # Add a visited location and extra mileage to the route
    def add_visit(self, request, distances, is_pickup):
        self.mileage += distances[request.lid, self.visited[-1]]

        if not is_pickup:
            self.visited.append(request.rid)
        else:
            self.visited.append(-request.rid)

    def back_to_depot(self, reqs, distances):
        self.mileage += distances[reqs[abs(self.visited[-1]) - 1].lid, 0]
        self.visited.append(0)

    def possible_addition(self, request, distances, max_dist, day, tools):
        req_tools = [t for t in tools if t.tid == request.tid]

        count = 0
        for t in req_tools:
            available = 0
            for d in range(day, day + request.stay + 1):
                if t.in_use[d - 1] == 0:
                    available += 1
            if available == len(range(day, day + request.stay + 1)):
                count += 1

        pickup = count >= request.no_tools

        return self.mileage + distances[self.visited[-1], request.lid] + distances[request.lid, 0] \
            <= max_dist and pickup

# Importing the required modules and classes
from gurobipy import Model, GRB, quicksum
from location import Location
from request import Request
from tool import Tool
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

# Initializing global variables with default values
DATASET = ""
NAME = ""
DAYS = 0
CAPACITY = 0
MAX_TRIP_DISTANCE = 0
DEPOT_COORDINATE = 0
VEHICLE_COST = 0
VEHICLE_DAY_COST = 0
DISTANCE_COST = 0
TOOLS = []
LOCATIONS = []
REQUESTS = []
VEHICLES = []
TOOL_DICTI = defaultdict(int)


# Function to read a tool from a string representation
def read_tool(tool):
    tid, size, max_no, cost = (int(part) for part in tool.split())
    return Tool(tid, size, max_no, cost)


# Function to read a coordinate from a string representation
def read_coordinate(coordinate):
    i, x, y = (int(part) for part in coordinate.split())
    return Location(x, y)


# Function to read a request from a string representation
def read_request(request):
    rid, lid, first, last, stay, tid, no_tools = (int(part) for part in request.split())
    return Request(rid, lid, first, last, stay, tid, no_tools)


# Function to calculate the distance between each pair of coordinates
def calc_distances():
    result = np.empty(shape=(len(LOCATIONS), len(LOCATIONS)))
    for i in range(len(LOCATIONS)):
        for j in range(i):
            result[i, j] = LOCATIONS[i].distance(LOCATIONS[j])
    return result


# Function to calculate the cost of a given distance
def distance_cost(distance2):
    return distance2 * DISTANCE_COST


# Function to plot all coordinates
def plot_all():
    # Plot all coordinates with a blue dot marker
    plt.plot([c.x for c in LOCATIONS], [c.y for c in LOCATIONS], 'bo')

    # Plot the first coordinate with a red circle marker
    plt.plot(LOCATIONS[0].x, LOCATIONS[0].y, 'ro')

    # Add a title and axis labels
    plt.title('Plot of Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the plot
    plt.grid()
    plt.show()


def assign_tools():
    for t2 in TOOLS:
        LOCATIONS[0].add_tool(t2)


# Function to read data from a file
def read_file(txt):
    # Extracting global variables from file
    split = "= "
    global DATASET
    DATASET = txt[0].split(split)[1]

    global NAME
    NAME = txt[1].split(split)[1]

    global DAYS
    DAYS = int(txt[3].split(split)[1])

    global CAPACITY
    CAPACITY = int(txt[4].split(split)[1])

    global MAX_TRIP_DISTANCE
    MAX_TRIP_DISTANCE = int(txt[5].split(split)[1])

    global VEHICLE_COST
    VEHICLE_COST = int(txt[8].split(split)[1])

    global VEHICLE_DAY_COST
    VEHICLE_DAY_COST = int(txt[9].split(split)[1])

    global DISTANCE_COST
    DISTANCE_COST = int(txt[10].split(split)[1])

    # Reading tools from file
    no_tools = int(txt[12].split(split)[1])
    for i in range(1, no_tools + 1):
        TOOLS.append(read_tool(txt[12 + i]))

    # Reading coordinates from file
    no_coordinates = int(txt[12 + no_tools + 2].split(split)[1])
    for i in range(1, no_coordinates + 1):
        LOCATIONS.append(read_coordinate(txt[12 + no_tools + 2 + i]))

    # Reading requests from file
    no_requests = int(txt[12 + no_tools + no_coordinates + 4].split(split)[1])
    for i in range(1, no_requests + 1):
        REQUESTS.append(read_request(txt[12 + no_tools + no_coordinates + 4 + i]))


def create_file(filename):
    with open(filename, 'w') as f:
        f.write(f'DATASET = {DATASET}\n')
        f.write(f'NAME = {NAME}\n\n')
        # f.write(f'MAX_NUMBER_OF_VEHICLES = {}\n')
        # f.write(f'NUMBER_OF_VEHICLE_DAYS = {}\n')
        # f.write(f'TOOL_USE = {}\n')
        # f.write(f'DISTANCE = {}\n')
        # f.write(f'COST = {}\n\n')
        #
        # for i in range(1, DAYS+1):
        #     f.write(f'DAY = {i}\n')
        #     f.write(f'NUMBER_OF_VEHICLES = {}\n')
        #     f.write(f'{}\n')


if __name__ == '__main__':
    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()

    # Display a file dialog and wait for the user to select a file
    file_path = filedialog.askopenfilename()

    # Read the contents of the file into a list of strings
    with open(file_path, 'r') as file:
        input_lines = [line.strip() for line in file]

    read_file(input_lines)
    distances = calc_distances()
    plot_all()

    # Set of customers
    customers = list(set([request.loc_id for request in REQUESTS]))
    # Set of tool kinds
    tool_kinds = []
    for tool in TOOLS:
        tool_kinds.append(tool.tid)
    # Number of tools of each kind available
    tool_dict = defaultdict(int)
    for item in TOOLS:
        tool_dict[item.tid] = item.max_no
    # Requests: dictionary where each key is a customer and each value is a list of requests made by that customer
    requests_dict = defaultdict(list)
    for request in REQUESTS:
        requests_dict[request.loc_id].append(request)

    # Time window for each request
    time_windows = defaultdict(list)
    for request in REQUESTS:
        time_windows[request.rid] = (request.first, request.last)
    # Capacity of each vehicle
    vehicle_capacity = CAPACITY
    # Maximum distance a vehicle can travel in one day
    max_distance = MAX_TRIP_DISTANCE
    # Number of days in the planning horizon
    days = list(range(1, DAYS + 1))

    # Create a dictionary to store the last delivery day for each tool and customer
    holding_days = defaultdict(int)
    for request in REQUESTS:
        holding_days[request.rid] += request.stay

    # Create a new Gurobi model
    m = Model()

    # Create decision variables
    # Binary variable indicating whether a vehicle is used on a given day
    use_vehicle = m.addVars(days, vtype=GRB.BINARY, name="use_vehicle")

    # Binary variable indicating whether a vehicle visits a customer on a given day
    visit_customer = m.addVars(customers, days, vtype=GRB.BINARY, name="visit_customer")

    # Binary variable indicating whether a vehicle picks up a tool of a given kind from a customer on a given day
    pickup = m.addVars(customers, tool_kinds, days, vtype=GRB.BINARY, name="pickup")

    # Binary variable indicating whether a vehicle delivers a tool of a given kind to a customer on a given day
    delivery = m.addVars(customers, tool_kinds, days, vtype=GRB.BINARY, name="delivery")

    # Continuous variable indicating the distance traveled by a vehicle on a given day
    distance = m.addVars(days, lb=0, ub=max_distance, vtype=GRB.CONTINUOUS, name="distance")

    # Integer variable indicating the number of tools used
    tools_used = m.addVars(tool_kinds, vtype=GRB.INTEGER, name="tools_used")

    for c in customers:
        for r in requests_dict[c]:
            for d in range(r.first, r.last + 1):
                m.addConstr(quicksum(pickup[c, r.tool_id, d + i] for i in range(r.stay)) >=
                            quicksum(delivery[c, r.tool_id, d + i] for i in range(r.stay)))

    # Add constraints and objective function...
    # # Each customer must be visited at most once per day:
    # for d in days:
    #     for c in customers:
    #         m.addConstr(quicksum(visit_customer[c, d] for d in days) <= 1)
    # Each tool kind must be picked up and delivered at most once per day:
    for d in days:
        for t in tool_kinds:
            m.addConstr(quicksum(pickup[c, t, d] for c in customers) <= 1)
            m.addConstr(quicksum(delivery[c, t, d] for c in customers) <= 1)

    # If a vehicle visits a customer on a given day, it must pick up or deliver at least one tool:
    for d in days:
        for c in customers:
            m.addConstr(visit_customer[c, d] <= quicksum(pickup[c, t, d] + delivery[c, t, d] for t in tool_kinds))

    # The capacity of a vehicle must not be exceeded on any day:
    for d in days:
        m.addConstr(quicksum(requests_dict[c][i].no_tools * (
                pickup[c, requests_dict[c][i].tool_id, d] - delivery[c, requests_dict[c][i].tool_id, d]) for c in
                             customers for i in range(len(requests_dict[c]))) <= vehicle_capacity)

    # If a vehicle is used on a given day, its total distance traveled must not exceed the maximum distance:
    for d in days:
        m.addConstr(distance[d] >= quicksum(
            distances[i][j] * visit_customer[c, d] * visit_customer[c_prime, d] for i in customers for j in
            customers for c in [i] for c_prime in [j]))
        m.addConstr(distance[d] <= max_distance * use_vehicle[d])

    for r in REQUESTS:
        for d in days:
            m.addConstr(quicksum(pickup[r.loc_id, k, t] for k in tool_kinds for t in range(r.first, r.last + 1))
                        + quicksum(delivery[r.loc_id, k, t] for k in tool_kinds for t in range(r.first, r.last + 1))
                        == 1, f"request_{r.rid}_time_window_day_{d}")

    # If a tool of a given kind is delivered to a customer on a given day,
    # it must be picked up exactly i days later:
    for r in requests_dict[c]:
        for d in range(r.first, r.last + 1):
            for i in range(r.stay):
                if d + i + r.stay <= r.last:
                    m.addConstr(delivery[c, r.tool_id, d] <= pickup[c, r.tool_id, d + i + r.stay])
                    m.addConstr(pickup[c, r.tool_id, d + i + r.stay] <= delivery[c, r.tool_id, d])

    m.setObjective(quicksum(distance[d] for d in days), GRB.MINIMIZE)
    m.setObjective(quicksum(use_vehicle[d] for d in days), GRB.MINIMIZE)

    m.setObjective(quicksum(tools_used[t] for t in tool_kinds), GRB.MINIMIZE)

    # Solve the model
    m.optimize()

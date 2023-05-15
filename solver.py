# Importing the required modules and classes
from itertools import combinations

from location import Location
from request import Request
from route import Route
from tool import Tool
from day import Day
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from gurobipy import *
import copy

from vehicle import Vehicle

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
SCHEDULE = []
TOOLS = []
ALL_TOOLS = []
LOCATIONS = []
REQUESTS = []
VEHICLES = []
DISTANCES = np.empty(0)


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

    for i in range(DAYS):
        SCHEDULE.append(Day())

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

    for t in TOOLS:
        for d in SCHEDULE:
            d.depot_tools.update({t.tid: t.max_no})

        for i in range(1, t.max_no + 1):
            tool = copy.deepcopy(t)
            tool.id = i
            ALL_TOOLS.append(tool)

    # Reading requests from file
    no_requests = int(txt[12 + no_tools + no_coordinates + 4].split(split)[1])
    for i in range(1, no_requests + 1):
        REQUESTS.append(read_request(txt[12 + no_tools + no_coordinates + 4 + i]))


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


# Function to read a tool from a string representation
def read_tool(tool):
    tid, size, max_no, cost = (int(part) for part in tool.split())
    return Tool(tid, size, max_no, cost)


# Function to read a coordinate from a string representation
def read_coordinate(coordinate):
    i, x, y = (int(part) for part in coordinate.split())
    return Location(i, x, y)


# Function to read a request from a string representation
def read_request(request):
    rid, lid, first, last, stay, tid, no_tools = (int(part) for part in request.split())
    req = Request(rid, lid, first, last, stay, tid, no_tools)
    VEHICLES.append(Vehicle(len(VEHICLES) + 1, DAYS))  # VEHICLES.append(Vehicle(rid))
    return req


# Function to create a distance matrix calculating the distances between
# pairs of coordinates and the distance from depot to the furthest location
def calc_distances():
    max_depot_dist = 0
    result = np.empty(shape=(len(LOCATIONS), len(LOCATIONS)))
    for i in range(len(LOCATIONS)):
        for j in range(len(LOCATIONS)):
            dist = LOCATIONS[i].distance(LOCATIONS[j])
            result[i, j] = dist

            if (i == 0 or j == 0) and dist > max_depot_dist:
                max_depot_dist = dist
    return result, max_depot_dist


# Function to calculate the total costs given a value of distance costs
def final_costs_distance():
    v_days = 0
    max_v = 0
    total_dist = 0

    for d in SCHEDULE:
        d.calc_mileage()
        total_dist += d.mileage

        v_days += len(d.routes)
        if len(d.routes) > max_v:
            max_v = len(d.routes)

    tool_cost = 0
    for t in ALL_TOOLS:
        if t.used:
            tool_cost += t.cost

    tot_costs = v_days * VEHICLE_DAY_COST + max_v * VEHICLE_COST + tool_cost

    return tot_costs, total_dist


def order_by_first():
    return sorted(REQUESTS, key=lambda r: (r.tid, r.first))


def schedule_requests_ILP(requests, tools):
    for r in requests:
        r.stay += 1

    # Create a new model
    m = Model("scheduling")

    # Create variables
    x = {}  # job r starts on machine t at day d
    for r in requests:
        for t in tools:
            for day in range(1, DAYS + 1):
                x[(r, t, day)] = m.addVar(vtype=GRB.BINARY, name=f"{day},{r.rid}")
                # y[(r, t, day)] = m.addVar(vtype=GRB.BINARY, name=f"y_r{r.rid}_t{t.tid}.{t.id}_d{day}")

    # Set objective: minimize the maximum lateness
    max_lateness = m.addVar(vtype=GRB.CONTINUOUS, name="max_lateness")
    m.addConstr(max_lateness <= 0, "max_lateness_non_positive")
    m.setObjective(max_lateness, GRB.MINIMIZE)

    # Add constraints
    for r in requests:
        # Each job must start exactly r.units number of times
        m.addConstr(quicksum(x[(r, t, day)] for t in tools for day in range(r.first, r.last + 1)) == r.units)
        if r.first > 1:
            m.addConstr(quicksum(x[(r, t, day)] for t in tools for day in range(1, r.first)) == 0)
        m.addConstr(quicksum(x[(r, t, day)] for t in tools for day in range(r.last + 1, DAYS + 1)) == 0)

        # If a job starts on a day, it must start on r.units number of machines
        for day in range(r.first, r.last + 1):
            for t in tools:
                m.addConstr(quicksum(x[(r, t_other, day)] for t_other in tools) >= r.units * x[(r, t, day)])

        # The maximum lateness must be greater than or equal to the lateness of each job
        for day in range(r.first, min(r.last + 1, DAYS - r.stay + 1)):
            m.addConstr(max_lateness >= (day + r.stay - (r.last + r.stay)) * x[(r, t, day)])

    for t in tools:
        for day in range(1, DAYS + 1):
            # Each machine can process at most one job per day
            m.addConstr(quicksum(x[(r, t, day)] for r in requests) <= 1)

            # If a job starts on a machine on a specific day, no other job can start on the same machine during its
            # processing time
            for r in requests:
                for r_other in requests:
                    if r_other != r:
                        m.addConstr(quicksum(x[(r, t, d)] for d in range(day, min(day + r.stay, DAYS + 1)))
                                    + quicksum(x[(r_other, t, d_other)] for d_other in
                                               range(max(day, r_other.first),
                                                     min(day + r.stay, r_other.last + 1))) <= 1)

    # Optimize model
    m.optimize()

    # Return the optimal schedule as a list of lists of requests per day
    days = [list() for d in range(1, DAYS + 1)]

    for v in m.getVars():
        if v.x > 0.5:
            entry = [int(value) for value in v.varName.split(',')]
            days[entry[0]].append(entry[1])

    return days


# Function that creates the output file
def create_file(filename, total_costs, total_distance):
    with open(filename, 'w') as f:
        f.write(f'DATASET = {DATASET}\n')
        f.write(f'NAME = {NAME}\n\n')

        max_vehicles = 0
        for d in SCHEDULE:
            if len(d.routes) > max_vehicles:
                max_vehicles = len(d.routes)
        f.write(f'MAX_NUMBER_OF_VEHICLES = {max_vehicles}\n')

        vehicle_days = 0
        for v in VEHICLES:
            vehicle_days += v.active_days

        f.write(f'NUMBER_OF_VEHICLE_DAYS = {vehicle_days}\n')

        tool_use = [0 for t in TOOLS]
        for t in ALL_TOOLS:
            if t.used:
                tool_use[t.tid - 1] += 1

        f.write(f'TOOL_USE = {" ".join(map(str, tool_use))}\n')
        f.write(f'DISTANCE = {int(total_distance)}\n')

        f.write(f'COST = {int(total_costs)}\n\n')

        for i in range(1, DAYS + 1):
            no_vehicles = len(SCHEDULE[i - 1].routes)
            if no_vehicles > 0:
                f.write(f'DAY = {i}\n')
                f.write(f'NUMBER_OF_VEHICLES = {no_vehicles}\n')
                for r in SCHEDULE[i - 1].routes:
                    f.write(f'{r.vid} R')
                    for v in r.visited:
                        f.write(f' {v}')
                    f.write('\n')
                f.write('\n')


if __name__ == '__main__':
    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()

    # Display a file dialog and wait for the user to select a file
    file_path = filedialog.askopenfilename()

    # Read the contents of the file into a list of strings
    with open(file_path, 'r') as file:
        input_lines = [line.strip() for line in file]

    # Processing input
    read_file(input_lines)
    DISTANCES, max_dist_depot = calc_distances()
    # plot_all()

    # Creating the schedule
    priorities = order_by_first()
    final = [list() for i in range(1, DAYS+1)]
    for t in TOOLS:
        req_lst = [r for r in priorities if r.tid == t.tid]
        tool_lst = [tool for tool in ALL_TOOLS if t.tid == tool.tid]

        result = schedule_requests_ILP(req_lst, tool_lst)

        for i, res in enumerate(result):
            for req in res:
                final[i-1].append(req)

    for f in final:
        print(f)
    # Creating the output
    # costs, total_dist = final_costs_distance()
    # create_file(file_path.split("/")[-1].split(".")[0]+"sol.txt", costs, total_dist)

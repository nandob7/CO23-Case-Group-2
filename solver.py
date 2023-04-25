# Importing the required modules and classes
from location import Location
from request import Request
from route import Route
from tool import Tool
from day import Day
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

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
PROCESS_ON_FIRST = [Day()]
TOOLS = []
ALL_TOOLS = []
LOCATIONS = []
REQUESTS = []
VEHICLES = [Vehicle(0)]


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
    PROCESS_ON_FIRST[first].requests.append(req)
    PROCESS_ON_FIRST[first + stay].requests.append(req)
    VEHICLES.append(Vehicle(rid))
    return req


# Function to calculate the distance between each pair of coordinates
def calc_distances():
    result = np.empty(shape=(len(LOCATIONS), len(LOCATIONS)))
    for i in range(len(LOCATIONS)):
        for j in range(len(LOCATIONS)):
            result[i, j] = LOCATIONS[i].distance(LOCATIONS[j])
    return result


# Function to calculate the cost of a given distance
def calculate_total_costs(distance_costs):
    result = distance_costs
    for v in VEHICLES:
        if v.active_days > 0:
            result += VEHICLE_COST + v.active_days * VEHICLE_DAY_COST

    for t in ALL_TOOLS:
        if t.used:
            result += t.cost

    return result


def create_schedule():
    schedule = list()
    total_costs = 0
    total_distance = 0
    for day in range(1, DAYS + 1):
        for r in PROCESS_ON_FIRST[day].requests:
            new_route = Route(day)
            if new_route.possible_addition(LOCATIONS[r.lid], distances, MAX_TRIP_DISTANCE):
                new_route.add_visit(r, distances, ALL_TOOLS)
                new_route.back_to_depot(REQUESTS, distances)
                PROCESS_ON_FIRST[day].routes.append(new_route)
                VEHICLES[r.rid].assign_route(new_route)

                total_distance += new_route.mileage
                total_costs += new_route.calculate_route_cost(DISTANCE_COST)
    return total_costs, total_distance


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

    for day in range(1, DAYS + 1):
        PROCESS_ON_FIRST.append(Day())

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
        LOCATIONS[0].stored_tools.update({t: t.max_no})
        for i in range(1, t.max_no + 1):
            ALL_TOOLS.append(t)

    # Reading requests from file
    no_requests = int(txt[12 + no_tools + no_coordinates + 4].split(split)[1])
    for i in range(1, no_requests + 1):
        REQUESTS.append(read_request(txt[12 + no_tools + no_coordinates + 4 + i]))


def create_file(filename, total_costs, total_distance):
    with open(filename, 'w') as f:
        f.write(f'DATASET = {DATASET}\n')
        f.write(f'NAME = {NAME}\n\n')
        f.write(f'MAX_NUMBER_OF_VEHICLES = {len(VEHICLES) - 1}\n')

        vehicle_days = 0
        for v in VEHICLES:
            vehicle_days += v.active_days

        f.write(f'NUMBER_OF_VEHICLE_DAYS = {vehicle_days}\n')

        tool_use = [0 for t in range(1, TOOLS[-1].tid + 1)]
        for t in ALL_TOOLS:
            if t.used:
                tool_use[t.tid - 1] += 1

        f.write(f'TOOL_USE = {" ".join(map(str, tool_use))}\n')
        f.write(f'DISTANCE = {total_distance}\n')

        f.write(f'COST = {total_costs}\n\n')

        for i in range(1,DAYS+1):
            no_vehicles = len(PROCESS_ON_FIRST[i].routes)
            if no_vehicles > 0:
                f.write(f'DAY = {i}\n')
                f.write(f'NUMBER_OF_VEHICLES = {no_vehicles}\n')
                for r in PROCESS_ON_FIRST[i].routes:
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

    read_file(input_lines)
    distances = calc_distances()
    plot_all()
    distance_costs, distance = create_schedule()
    costs = calculate_total_costs(distance_costs)
    create_file("test.txt", costs, distance)

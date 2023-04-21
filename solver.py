# Importing the required modules and classes
from location import Location
from request import Request
from tool import Tool
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

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
PROCESS_ON_FIRST = []
TOOLS = []
LOCATIONS = []
REQUESTS = []
VEHICLES = []


# Function to read a tool from a string representation
def read_tool(tool):
    tid, size, max_no, cost = (int(part) for part in tool.split())
    return Tool(tid, size, max_no, cost)


def set_tools_depot():
    for t in TOOLS:
        LOCATIONS[0].stored_tools.update({t, t.max_no})


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
    return req


# Function to calculate the distance between each pair of coordinates
def calc_distances():
    result = np.empty(shape=(len(LOCATIONS), len(LOCATIONS)))
    for i in range(len(LOCATIONS)):
        for j in range(len(LOCATIONS)):
            result[i, j] = LOCATIONS[i].distance(LOCATIONS[j])
    return result


# Function to calculate the cost of a given distance
def distance_cost(distance):
    return distance * DISTANCE_COST


def createSchedule():
    schedule = list()
    for day in range(1, DAYS + 1):
        routes = [day]
        # for r in REQUESTS:
        #     if r.first <= day:
                # if .possible_addition(r.lid):

        return


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

    set_tools_depot()

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
        # f.write(f'MAX_NUMBER_OF_VEHICLES = {len(VEHICLES)}\n')
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
    create_file("test.txt")

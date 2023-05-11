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
SCHEDULE = []
TOOLS = []
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
    # PROCESS_ON_FIRST[first].requests.append(req)
    # PROCESS_ON_FIRST[first + stay].requests.append(req)
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
    min_dep_t = {t.tid: float('inf') for t in TOOLS}
    total_dist = 0

    for d in SCHEDULE:
        d.calc_mileage()
        total_dist += d.mileage

        v_days += len(d.routes)
        if len(d.routes) > max_v:
            max_v = len(d.routes)

        for t in TOOLS:
            if d.depot_tools.get(t.tid) < min_dep_t.get(t.tid):
                min_dep_t[t.tid] = d.depot_tools.get(t.tid)

    tool_cost = 0
    for t in TOOLS:
        t.used = t.max_no - min_dep_t.get(t.tid)
        tool_cost += t.cost * (t.max_no - min_dep_t.get(t.tid))

    tot_costs = v_days * VEHICLE_DAY_COST + max_v * VEHICLE_COST + tool_cost

    return tot_costs, total_dist


def request_prios(max_dist_depot):
    max_twindows = list(0 for r in TOOLS)
    max_stays = list(0 for r in TOOLS)
    for r in REQUESTS:
        if len(r.twindow) > max_twindows[r.tid - 1]:
            max_twindows[r.tid - 1] = len(r.twindow)
        if r.stay > max_stays[r.tid - 1]:
            max_stays[r.tid - 1] = r.stay

    for r in REQUESTS:
        r.priority_calc(max_twindows, max_stays, TOOLS, DISTANCES, max_dist_depot)

    return sorted(REQUESTS, key=lambda r: r.priority, reverse=True)


# Function that calculates the potential added costs of adding a request (and route) to
# a certain day in the horizon, can be given v_use and t_use to see if
# any formerly unused vehicles or tools are needed and considered in the added costs
def cost_if_added(day, request, route, t_use=None, v_use=0):
    travel_costs = (day.mileage + route.mileage) * DISTANCE_COST
    no_routes = len(day.routes) + 1

    v_no_cost = 0
    if no_routes > v_use > 0:
        v_no_cost = VEHICLE_COST * (no_routes - v_use)

    v_cost = no_routes * VEHICLE_DAY_COST + v_no_cost

    t_cost = 0
    if t_use is not None and day.depot_tools.get(request.tid) - request.no_tools \
            < t_use.get(request.tid):
        t_cost += (t_use.get(request.tid) - day.depot_tools.get(request.tid)
                   - request.no_tools) * TOOLS[request.tid - 1].cost

    print(t_cost)
    return travel_costs + v_cost + t_cost


# Function that returns a dictionary with the least unused tools for each tool
# over the time horizon -> tracks most simultaneously used tools for costs
def min_available_tools():
    result = {t.tid: float('inf') for t in TOOLS}
    for d in SCHEDULE:
        for t in TOOLS:
            if d.depot_tools.get(t.tid) < result.get(t.tid):
                result.update({t.tid: d.depot_tools.get(t.tid)})

    return result


# Function that finds the optimal day (cheapest) to schedule a request and its pickup
# given a time window for the request based on the current schedule
def find_opt_day(request, available_days):
    best = None
    min_cost = float('inf')

    # For loop over the time window to find the minimal added costs
    for day in available_days:
        end = day + request.stay
        delivery = Route(day)
        pickup = Route(end)

        # Checks if the request is possible to schedule on the day considering constraints
        if delivery.possible_addition(request, DISTANCES, MAX_TRIP_DISTANCE,
                                      SCHEDULE, day):
            delivery.add_visit(request, DISTANCES)
            # Gelijk terug naar depot voor simpel begin
            delivery.back_to_depot(REQUESTS, DISTANCES)
            old_d_cost = SCHEDULE[day - 1].mileage * DISTANCE_COST + \
                         len(SCHEDULE[day - 1].routes) * VEHICLE_DAY_COST
            added_cost = cost_if_added(SCHEDULE[day - 1], request, delivery,
                                       min_available_tools()) - old_d_cost
            pickup.add_visit(request, DISTANCES)
            # Gelijk terug naar depot voor simpel begin
            pickup.back_to_depot(REQUESTS, DISTANCES)
            old_p_cost = SCHEDULE[end - 1].mileage * DISTANCE_COST + \
                         len(SCHEDULE[end - 1].routes) * VEHICLE_DAY_COST
            added_cost += cost_if_added(SCHEDULE[end - 1], request,
                                        pickup) - old_p_cost
            if added_cost < min_cost:
                min_cost = added_cost
                best = day

    return best, min_cost


# Schedules a request either delivery or pickup given the chosen day.
def schedule_request(day, request):
    is_delivery = request.pickup is None
    new_route = Route(day)
    new_route.add_visit(request, DISTANCES)
    new_route.back_to_depot(REQUESTS, DISTANCES)

    SCHEDULE[day - 1].schedule(request, new_route)

    # Updates request status and tools use on the day and duration of stay
    if is_delivery:
        request.deliver(day)
        for d in range(request.stay):
            SCHEDULE[day + d].depot_tools[request.tid] -= request.no_tools

    # Loop over all vehicles and assigns the route to the first available then breaks.
    for v in VEHICLES:
        if v.active[day - 1] == 0:
            v.assign_route(new_route)

            # Assigns vid to the request for tracking and backtracking
            if is_delivery:
                request.vid[0] = v.vid
            else:
                request.vid[1] = v.vid
            break


# Unschedule a request given the day, for backtracking purposes
def unschedule_request(day, request):
    for v in VEHICLES:
        if v.vid in request.vid:
            v.active[day - 1] = 0

    request.vid = [0, 0]

    SCHEDULE[request.pickup - 1].unschedule(request)
    SCHEDULE[day - 1].unschedule(request)

    request.pickup = None


# Recursive function that loops over all days and tries to schedule the given requests in
# order on the best day and
def plan_requests(requests):
    if not requests:
        return True

    r = requests[0]
    best_day, _ = find_opt_day(r, r.twindow)

    if best_day is not None:
        schedule_request(best_day, r)
        schedule_request(r.pickup, r)

        if plan_requests(requests[1:]):
            return True

        # Backtrack if a feasible solution is not found
        unschedule_request(best_day, r)
        r.twindow.remove(best_day)

    # Time windows gaan naar 0 en daarom kan hij nog geen ander schedule
    # vinden na backtracken vgm
    if r.pickup is not None:
        day = r.pickup - r.stay
        unschedule_request(day, r)
        r.twindow.remove(day)

        if plan_requests(requests):
            return True

    return False


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
        for t in TOOLS:
            if t.used:
                tool_use[t.tid - 1] += t.used

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
    plot_all()
    # distance_costs, distance = create_schedule()

    # Creating the schedule
    priorities = request_prios(max_dist_depot)
    plan_requests(priorities)

    # Creating the output
    costs, total_dist = final_costs_distance()
    create_file("test.txt", costs, total_dist)

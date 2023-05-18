# Importing the required modules and classes
from day import Day
from gurobipy import *
from location import Location
from request import Request
from route import Route
from tkinter import filedialog
from tool import Tool
from vehicle import Vehicle
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import tkinter as tk

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
    return Tool(tid, size, max_no, cost, DAYS)


# Function to read a coordinate from a string representation
def read_coordinate(coordinate):
    i, x, y = (int(part) for part in coordinate.split())
    return Location(i, x, y)


# Function to read a request from a string representation
def read_request(request):
    rid, lid, first, last, stay, tid, no_tools = (int(part) for part in request.split())
    req = Request(rid, lid, first, last, stay, tid, no_tools)
    VEHICLES.append(Vehicle(len(VEHICLES) + 1, DAYS))
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
    max_v = 0
    mileage = 0

    for d in SCHEDULE:
        d.calc_mileage()
        mileage += d.mileage

        if len(d.routes) > max_v:
            max_v = len(d.routes)

    vehicle_days = 0
    for v in VEHICLES:
        vehicle_days += sum(v.active)

    tool_cost = 0
    used_tools = [t for t in ALL_TOOLS if sum(t.in_use) > 0]
    for t in used_tools:
        tool_cost += t.cost

    tot_costs = vehicle_days * VEHICLE_DAY_COST + max_v * VEHICLE_COST + tool_cost + mileage * DISTANCE_COST

    return tot_costs, mileage


def order_by_first():
    return sorted(REQUESTS, key=lambda r: (r.tid, r.first))


# Finds a feasible efficient routing for day of the daily request schedule
def daily_routing_ILP(curr_schedule):
    # Problem data
    num_locations = len(curr_schedule) + 1
    num_vehicles = len(curr_schedule)
    c2 = VEHICLE_COST + VEHICLE_DAY_COST

    # List indicating if a customer is a pickup (1) or delivery (0)
    p = [0]
    for rid in curr_schedule:
        if REQUESTS[abs(rid) - 1].pickup is None:
            p.append(0)
        else:
            p.append(1)

    # Now we can build the locations list
    locations_lst = [*range(num_locations)]

    # To compute the demands, we'll need to go through each request
    demands = [0]  # Start with 0 demand for the depot
    for rid in curr_schedule:
        # The demand for each request is TOOL[request.tid].size * request.units
        demand = TOOLS[REQUESTS[abs(rid) - 1].tid - 1].size * REQUESTS[abs(rid) - 1].units
        demands.append(demand)

    # Finally, we can extract the coordinates
    dep = LOCATIONS[DEPOT_COORDINATE]
    coordinates2 = [(dep.x, dep.y)]
    for rid in curr_schedule:
        # Every coordinate for a request can be found in LOCATIONS[request.lid]
        x_coordinate = LOCATIONS[REQUESTS[abs(rid) - 1].lid].x
        y_coordinate = LOCATIONS[REQUESTS[abs(rid) - 1].lid].y

        coordinates2.append((x_coordinate, y_coordinate))

    # Distance matrix
    distances = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            x1, y1 = coordinates2[i]
            x2, y2 = coordinates2[j]

            distances[i][j] = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 500)

    # Create model
    m = Model("cvrp")

    # Create variables
    x = {}
    u = {}
    y = {}

    for v in range(num_vehicles):
        y[v] = m.addVar(vtype=GRB.BINARY)
        for i in locations_lst:
            if i > 0:
                u[i] = m.addVar(vtype=GRB.INTEGER)
            for j in locations_lst:
                if i != j:
                    x[i, j, v] = m.addVar(vtype=GRB.BINARY)

    m.setObjective(
        quicksum(
            distances[i][j] * DISTANCE_COST * x[i, j, v] for i in locations_lst for j in locations_lst if i != j for v
            in
            range(num_vehicles)) +
        c2 * quicksum(y[v] for v in range(num_vehicles)),
        GRB.MINIMIZE)

    # Create constraints
    for v in range(num_vehicles):
        for i in locations_lst:
            for j in locations_lst:
                if i != j:
                    m.addConstr(y[v] >= x[i, j, v])

    # Add capacity constraints
    for v in range(num_vehicles):
        m.addConstr(
            quicksum(demands[i] * x[i, j, v] for i in locations_lst for j in locations_lst if i != j) <= CAPACITY)

    # Add distance constraints
    dist_constr = []
    for v in range(num_vehicles):
        dist_constr.append(m.addConstr(
            quicksum(distances[i][j] * x[i, j, v] for i in locations_lst for j in locations_lst if
                     i != j) <= MAX_TRIP_DISTANCE))

    # Each location should be visited exactly once
    for i in locations_lst[1:]:
        m.addConstr(quicksum(x[i, j, v] for j in locations_lst if i != j for v in range(num_vehicles)) == 1)

    # MTZ constraints
    for i in locations_lst[1:]:
        for j in locations_lst[1:]:
            if i != j:
                for v in range(num_vehicles):
                    m.addConstr(u[i] - u[j] + num_locations * x[i, j, v] <= num_locations - 1)

    # Flow constraints
    for v in range(num_vehicles):
        for j in locations_lst[1:]:
            m.addConstr(
                quicksum(x[i, j, v] for i in locations_lst if i != j) == quicksum(
                    x[j, k, v] for k in locations_lst if k != j))

    # Depot constraints
    for v in range(num_vehicles):
        m.addConstr(quicksum(x[0, j, v] for j in locations_lst[1:]) == y[v])
        m.addConstr(quicksum(x[i, 0, v] for i in locations_lst[1:]) == y[v])

    # # Optimize model and limit runtime and distance from optimal solution
    m.setParam('Presolve', 2)  # Use aggressive presolve
    m.setParam('Cuts', 2)  # Generate aggressive cuts
    m.setParam('Heuristics', 0.1)  # Spend 10% of time on heuristics
    m.setParam('VarBranch', 0)  # Use pseudocost variable branching
    m.setParam('Threads', 4)  # Use 4 threads
    m.setParam('NodeMethod', 1)  # Use dual simplex for node relaxations
    m.setParam('NodeSelection', 1)  # Select node with best bound

    # for c in dist_constr:
    #     c.UB = MAX_TRIP_DISTANCE
    m.setParam('MIPGap', 0.01)

    m.optimize()

    # Print the optimal solution
    routes = []
    for v in range(num_vehicles):
        route = []
        for i in locations_lst:
            for j in locations_lst:
                if i != j and x[i, j, v].x > 0.5:
                    if i != 0:
                        if p[i] == 1:
                            route.append(- curr_schedule[i - 1])
                        else:
                            route.append(curr_schedule[i - 1])
                    break
        if len(route) > 0:
            routes.append(route)

    return routes


# Schedules route of requests on a day
def schedule_requests(day, requests):
    new_route = Route(day)
    for rid in requests:
        r = REQUESTS[abs(rid) - 1]
        new_route.add_visit(r, DISTANCES, rid < 0, REQUESTS)

    new_route.back_to_depot(REQUESTS, DISTANCES)

    SCHEDULE[day - 1].schedule(new_route)

    # Updates tools use on the day and duration of stay
    for rid in requests:
        if rid > 0:
            r = REQUESTS[abs(rid) - 1]

            tools = [t for t in ALL_TOOLS if t.tid == r.tid]
            assigned_tools = []
            for t in tools:
                if sum(t.in_use[day - 1: day + r.stay]) == 0:
                    assigned_tools.append(t)
                    for d in range(day - 1, day + r.stay):
                        t.in_use[d] = True
                    if len(assigned_tools) == r.units:
                        break

    # Loop over all vehicles and assigns the route to the first available then breaks.)
    for v in VEHICLES:
        if v.active[day - 1] == 0:
            v.assign_route(new_route)
            break


def schedule_request(day, request):
    is_delivery = request.pickup is None
    new_route = Route(day)
    new_route.add_visit(request, DISTANCES, not is_delivery, REQUESTS)
    new_route.back_to_depot(REQUESTS, DISTANCES)

    SCHEDULE[day - 1].schedule(new_route)

    # Updates request status and tools use on the day and duration of stay
    if is_delivery:
        request.deliver(day)
        tools = [t for t in ALL_TOOLS if t.tid == request.tid]
        assigned_tools = []
        for t in tools:
            if sum(t.in_use[day - 1: day + request.stay]) == 0:
                assigned_tools.append(t)
                for d in range(day - 1, day + request.stay):
                    t.in_use[d] = True
                if len(assigned_tools) == request.units:
                    break

    # Loop over all vehicles and assigns the route to the first available then breaks.)
    for v in VEHICLES:
        if v.active[day - 1] == 0:
            v.assign_route(new_route)
            break


def plan_schedule(sorted_requests, plan):
    schedule = [list() for i in range(1, DAYS + 1)]
    for t in TOOLS:
        req_lst = [r for r in sorted_requests if r.tid == t.tid]
        tool_lst = [tool for tool in ALL_TOOLS if t.tid == tool.tid]

        result = daily_schedule_ILP(req_lst, tool_lst)

        for i, res in enumerate(result):
            for req in res:
                schedule[i].append(req)

    for i, day in enumerate(schedule):
        schedule[i] = list(set(day))

    if plan:
        for i, day in enumerate(schedule):
            for rid in day:
                r = REQUESTS[rid - 1]
                schedule_request(i + 1, r)
                schedule_request(r.pickup, r)

    for i, d in enumerate(schedule):
        for rid in d:
            if rid > 0:
                r = REQUESTS[rid - 1]
                schedule[i + r.stay].append(-r.rid)

    return schedule


def daily_schedule_ILP(requests, tools):
    # Add 1 to length of stay as we are not considering picking up and delivering to new request on the same day
    for r in requests:
        r.stay += 1

    # Create a new model
    m = Model("scheduling")

    # Create variables
    x = {}  # job r starts on machine t at day d
    y = {}  # machine t used
    for r in requests:
        for t in tools:
            for day in range(1, DAYS + 1):
                if day in range(r.first, r.last + 1):
                    x[(r, t, day)] = m.addVar(vtype=GRB.BINARY, name=f"{day},{r.rid}")

    for t in tools:
        y[t] = m.addVar(vtype=GRB.BINARY, name=f"y_{t}")

    # Set objective: minimize no. tools used
    m.setObjective(quicksum(y[t] for t in tools), GRB.MINIMIZE)

    # Add constraints
    # Max lateness has to be <= 0
    max_lateness = m.addVar(vtype=GRB.CONTINUOUS, name="max_lateness")
    m.addConstr(max_lateness <= 0, "max_lateness_non_positive")

    for r in requests:
        # Each job must start exactly r.units number of times
        m.addConstr(quicksum(x[(r, t, day)] for t in tools for day in range(r.first, r.last + 1)) == r.units)

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
            m.addConstr(quicksum(x[(r, t, day)] for r in requests if day in range(r.first, r.last + 1)) <= y[t])

            # If a job starts on a machine on a specific day, no other job can start on the same machine during its
            # processing time
            for r in requests:
                for r_other in requests:
                    if r_other != r:
                        m.addConstr(quicksum(x[(r, t, d)] for d in range(day, min(day + r.stay, r.last + 1)) if
                                             day in range(r.first, r.last + 1))
                                    + quicksum(x[(r_other, t, d_other)] for d_other in
                                               range(max(day, r_other.first),
                                                     min(day + r.stay, r_other.last + 1))) <= 1)

    # Optimize model
    m.setParam('Presolve', 2)  # Use aggressive presolve
    m.setParam('Cuts', 2)  # Generate aggressive cuts
    m.setParam('Heuristics', 0.1)  # Spend 10% of time on heuristics
    m.setParam('VarBranch', 0)  # Use pseudocost variable branching
    m.setParam('Threads', 4)  # Use 4 threads
    m.setParam('NodeMethod', 1)  # Use dual simplex for node relaxations
    m.setParam('NodeSelection', 1)  # Select node with best bound

    m.optimize()

    # Return the optimal schedule as a list of lists of requests per day
    days = [list() for d in range(1, DAYS + 1)]

    for r in requests:
        r.stay -= 1

    for v in m.getVars():
        if v.x > 0.5 and not v.varName.startswith("y_"):
            entry = [int(value) for value in v.varName.split(',')]
            days[entry[0] - 1].append(entry[1])

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
            vehicle_days += sum(v.active)

        f.write(f'NUMBER_OF_VEHICLE_DAYS = {vehicle_days}\n')

        tool_use = [0 for t in TOOLS]
        for t in ALL_TOOLS:
            if sum(t.in_use) > 0:
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
    instance = file_path.split("/")[-1].split(".")[0]

    # Record the starting time
    start_time = time.time()

    # Read the contents of the file into a list of strings
    with open(file_path, 'r') as file:
        input_lines = [line.strip() for line in file]

    # Processing input
    read_file(input_lines)
    DISTANCES, max_dist_depot = calc_distances()
    # plot_all()

    # Order the requests by tool id and release date
    priorities = order_by_first()

    # Create a daily schedule of requests
    schedule = plan_schedule(priorities, plan=False)

    # Find routes for each day of the daily request schedule
    daily_routes = []
    for d in schedule:
        daily_routes.append(daily_routing_ILP(d))

    # Plan the routes for each day
    for i, day in enumerate(daily_routes):
        for routes in day:
            schedule_requests(i + 1, routes)

    # Creating the output
    costs, total_dist = final_costs_distance()
    create_file(instance + "sol.txt", costs, total_dist)

    #########################################################
    # Merge vehicle routes if sum of mileage < MAX TRIP DISTANCE
    #########################################################

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Print the runtime in seconds
    print(f"\nInstance: {instance}")
    print(f"Runtime: {elapsed_time:.2f}s")

import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    # Locations coordinates (depot is at index 0)
    data['locations'] = [
        (40, 50),  # depot
        (45, 68),  # customer 1
        (45, 70),  # customer 2
        (42, 66),  # customer 3
        (42, 68),  # customer 4
        (42, 65),  # customer 5
        (40, 69),  # customer 6
        (40, 66),  # customer 7
        (38, 68),  # customer 8
        (38, 70),  # customer 9
        (35, 66),  # customer 10
        (35, 69),  # customer 11
        (25, 85),  # customer 12
    ]
    
    # Number of vehicles
    data['num_vehicles'] = 3
    
    # Vehicle capacities - adding capacity constraints
    data['vehicle_capacities'] = [50, 50, 50]
    
    # Depot node index
    data['depot'] = 0
    
    # Demands for each location
    data['demands'] = [0, 10, 15, 5, 8, 12, 10, 7, 9, 11, 14, 6, 13]
    
    # Time windows for each location in minutes from start of day
    # Format: (earliest service time, latest service time)
    data['time_windows'] = [
        (0, 1000),       # depot
        (100, 400),      # customer 1
        (200, 400),      # customer 2
        (100, 300),      # customer 3
        (150, 250),      # customer 4
        (100, 400),      # customer 5
        (250, 500),      # customer 6
        (200, 450),      # customer 7
        (150, 350),      # customer 8
        (300, 500),      # customer 9
        (250, 450),      # customer 10
        (200, 350),      # customer 11
        (350, 650),      # customer 12
    ]
    
    # Service time in minutes at each location
    data['service_times'] = [
        0,  # depot
        20, # customer 1
        15, # customer 2
        30, # customer 3
        20, # customer 4
        25, # customer 5
        15, # customer 6
        20, # customer 7
        25, # customer 8
        15, # customer 9
        20, # customer 10
        30, # customer 11
        20, # customer 12
    ]
    
    return data

def compute_euclidean_distance_matrix(locations):
    """Creates callback to return the distance and time between two points."""
    distances = {}
    for from_node in range(len(locations)):
        distances[from_node] = {}
        for to_node in range(len(locations)):
            if from_node == to_node:
                distances[from_node][to_node] = 0
            else:
                # Calculate Euclidean distance and convert to travel time (assuming 1 unit = 1 minute)
                x1, y1 = locations[from_node]
                x2, y2 = locations[to_node]
                distances[from_node][to_node] = int(np.hypot(x1 - x2, y1 - y2))
    return distances

def get_solution(data, manager, routing, solution):
    """Prints solution on console and returns routes for visualization."""
    routes = []
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id}:\n'
        route = []
        route_time = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            
            time_var = time_dimension.CumulVar(index)
            plan_output += f' {node_index} Time({solution.Min(time_var)},{solution.Max(time_var)}) -> '
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_time += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        time_var = time_dimension.CumulVar(index)
        plan_output += f'{node_index} Time({solution.Min(time_var)},{solution.Max(time_var)})\n'
        plan_output += f'Time of the route: {route_time}min\n\n'
        
        print(plan_output)
        routes.append(route)
        total_time += route_time
    
    print(f'Total time of all routes: {total_time}min')
    return routes

def plot_solution(data, routes):
    """Plots the solution."""
    plt.figure(figsize=(15, 10))
    
    # Plot locations
    locations = data['locations']
    plt.scatter([loc[0] for loc in locations[1:]], [loc[1] for loc in locations[1:]], 
                c='blue', s=100, label='Customers')
    plt.scatter(locations[0][0], locations[0][1], c='red', s=200, marker='*', label='Depot')
    
    # Plot routes
    colors = ['darkgreen', 'purple', 'orange', 'brown', 'gray']
    
    for i, route in enumerate(routes):
        if len(route) > 2:  # Only plot non-empty routes
            route_x = [locations[j][0] for j in route]
            route_y = [locations[j][1] for j in route]
            plt.plot(route_x, route_y, c=colors[i % len(colors)], linewidth=2, alpha=0.7, label=f'Vehicle {i}')
    
    # Add time window labels
    for i, loc in enumerate(locations):
        earliest, latest = data['time_windows'][i]
        label = f"{i}: [{earliest}-{latest}]"
        plt.annotate(label, (loc[0], loc[1]), xytext=(5, 5), textcoords='offset points')
    
    plt.title('Vehicle Routing Solution with Time Windows')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Display the plot instead of just saving it
    plt.show()
    
    # Save the plot as an image
    plt.savefig('vrptw_solution.png')

def solve_vrptw():
    """Solve the Vehicle Routing Problem with Time Windows."""
    data = create_data_model()
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data['locations']),
        data['num_vehicles'],
        data['depot'])
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Get the distance matrix
    distance_matrix = compute_euclidean_distance_matrix(data['locations'])
    
    # Create and register transit callback
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node] + data['service_times'][from_node]
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    
    # Add Time Window constraints
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        # Allow waiting time at locations
        1000,  # max slack time
        2000,  # max time per vehicle
        False,  # Don't force start cumul to zero
        time)
    
    time_dimension = routing.GetDimensionOrDie(time)
    
    # Add time window constraints for each location except depot
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    
    # Add time window constraints for depot: either at the start or at the end
    index = manager.NodeToIndex(data['depot'])
    depot_time_window = data['time_windows'][data['depot']]
    time_dimension.CumulVar(index).SetRange(depot_time_window[0], depot_time_window[1])
    
    # Instantiate route start and end times to produce feasible times
    for vehicle_id in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(vehicle_id)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(vehicle_id)))
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Additional metaheuristics to improve solution
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30  # Increase time limit
    
    # Use threads for faster computation
    search_parameters.log_search = True  
    
    # Add penalties for unassigned locations
    for node in range(1, len(data['locations'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], 1000)
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        routes = get_solution(data, manager, routing, solution)
        plot_solution(data, routes)
        print("Solution found!")
        return True
    else:
        print("No solution found!")
        return False

if __name__ == "__main__":
    solve_vrptw()

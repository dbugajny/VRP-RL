from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from environment import Environment


"""
Link to OR-Tools documentations: https://developers.google.com/optimization/routing/cvrp
OR-Tools are used to calculate optimal (or near-optimal) solution
"""

def create_data_model(env: Environment, idx: int):
    n_locations = len(env.demands[idx])

    # It is assumed that there is one vehicle that can go to the depot as many times as needed.
    # In the worst case scenario, the vehicle will return to the depot after visiting each node.
    data = {
        "distance_matrix": create_distance_matrix(env.locations[idx]),
        "demands": env.demands[idx].numpy().astype(int).tolist(),
        "vehicle_capacities": [int(env.capacity[idx].numpy())] * n_locations,
        "num_vehicles": n_locations,
        "depot": 0,
    }
    return data


def distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def create_distance_matrix(locations):
    mat = np.zeros((5, 5))

    for i in range(len(locations)):
        for j in range(i+1, len(locations)):
            x1 = locations[i][0]
            x2 = locations[j][0]
            y1 = locations[i][1]
            y2 = locations[j][1]

            dist = distance(x1, x2, y1, y2)
            mat[i, j] = dist
            mat[j, i] = dist

    return mat.astype(int).tolist()


def get_solution_results(data, manager, routing, solution):
    total_distance = 0
    total_load = 0
    route_all = []

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        route = []

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        route.append(manager.IndexToNode(index))
        total_distance += route_distance
        total_load += route_load
        route_all.append(route)

    return total_distance, total_load, route_all


def or_tools_solve(data):
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data["vehicle_capacities"],
        True,
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(1)

    solution = routing.SolveWithParameters(search_parameters)

    return get_solution_results(data, manager, routing, solution)

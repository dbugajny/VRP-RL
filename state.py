import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

Coordinates = namedtuple("Coordinates", ["x", "y"])


class Location:
    def __init__(self, coordinates: Coordinates, is_depot: bool, demand: int | None = None) -> None:
        self.coordinates: Coordinates = coordinates
        self.is_depot: bool = is_depot
        self.demand: int | None = demand


class Vehicle:
    def __init__(self, coordinates: Coordinates, max_capacity: int) -> None:
        self.coordinates: Coordinates = coordinates
        self.max_capacity: int = max_capacity
        self.current_capacity: int = max_capacity


class State:
    def __init__(self, n_locations: int, max_demand: int, max_capacity: int) -> None:
        self.locations = [Location((Coordinates(np.random.uniform(-1, 1), np.random.uniform(-1, 1))), True)]

        for _ in range(n_locations - 1):
            self.locations.append(
                Location(
                    Coordinates(np.random.uniform(-1, 1), np.random.uniform(-1, 1)),
                    False,
                    np.random.randint(1, max_demand),
                )
            )

        self.vehicle: Vehicle = Vehicle(self.locations[0].coordinates, max_capacity)

        self.summed_cost: int = 0
        self.vehicle_history: list[Coordinates] = [self.vehicle.coordinates]

    def update_state(self, next_node: int) -> None:
        self.summed_cost += self.get_travel_cost(self.vehicle.coordinates, self.locations[next_node].coordinates)

        if self.locations[next_node].is_depot:
            self.vehicle.current_capacity = self.vehicle.max_capacity
        else:
            self.vehicle.current_capacity, self.locations[next_node].demand = max(
                0, self.vehicle.current_capacity - self.locations[next_node].demand
            ), max(0, self.locations[next_node].demand - self.vehicle.current_capacity)
        self.vehicle.coordinates = self.locations[next_node].coordinates
        self.vehicle_history.append(self.vehicle.coordinates)

    def are_all_demands_satisfied(self) -> bool:
        for location in self.locations[1:]:
            if location.demand > 0:
                return False
        return True

    @staticmethod
    def get_travel_cost(start_coordinates: Coordinates, end_coordinates: Coordinates):
        return np.sqrt((start_coordinates.x - end_coordinates.x) ** 2 + (start_coordinates.y - end_coordinates.y) ** 2)

    def __str__(self) -> str:
        string = f"Vehicle: {self.vehicle.coordinates}: {self.vehicle.current_capacity}\n"
        string += "Locations:\n"

        for location in self.locations:
            string += f"{location.coordinates}: {location.demand}\n"

        return string

    def visualize_state(self):
        fig, ax = plt.subplots()
        ax.scatter(self.vehicle.coordinates[0], self.vehicle.coordinates[1], s=100, facecolor="none", edgecolors="red")

        for location in self.locations:
            if location.is_depot:
                ax.scatter(location.coordinates[0], location.coordinates[1], color="green")
            else:
                ax.scatter(location.coordinates[0], location.coordinates[1], color="blue")
                ax.annotate(f"{location.demand}", (location.coordinates[0] + 0.02, location.coordinates[1] + 0.02))

        for i in range(len(self.vehicle_history) - 1):
            ax.plot(
                [self.vehicle_history[i][0], self.vehicle_history[i + 1][0]],
                [self.vehicle_history[i][1], self.vehicle_history[i + 1][1]],
            )

        plt.title(f"Current capacity: {self.vehicle.current_capacity}, Summed cost: {self.summed_cost: .2f}")
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.show()


class State2:
    def __init__(self, n_locations: int, max_demand: int, max_capacity: int) -> None:
        self.locations = np.random.uniform(-1, 1, size=(n_locations, 2))
        self.demands = np.random.randint(1, max_demand, size=n_locations)
        self.demands[0] = 0

        self.vehicle_position = self.locations[0]
        self.idx_position = 0
        self.max_capacity = max_capacity
        self.current_capacity = max_capacity

        self.summed_cost: float = 0
        self.vehicle_history: list = [self.vehicle_position]

    def update_state(self, next_node: int) -> None:
        self.summed_cost += self.get_travel_cost(self.vehicle_position, self.locations[next_node])

        if next_node == 0:
            self.current_capacity = self.max_capacity
        else:
            self.current_capacity, self.demands[next_node] = np.maximum(
                0, self.current_capacity - self.demands[next_node]
            ), np.maximum(0, self.demands[next_node] - self.current_capacity)
        self.vehicle_position = self.locations[next_node]
        self.idx_position = next_node
        self.vehicle_history.append(self.vehicle_position)

    def are_all_demands_satisfied(self) -> bool:
        return np.all(self.demands == 0)

    @staticmethod
    def get_travel_cost(start_position, end_position):
        return np.sqrt((start_position[0] - end_position[0]) ** 2 + (start_position[1] - end_position[1]) ** 2)

    def visualize_state(self):
        fig, ax = plt.subplots()
        ax.scatter(self.vehicle_position[0], self.vehicle_position[1], s=100, facecolor="none", edgecolors="red")

        ax.scatter(self.locations[0][0], self.locations[0][1], color="green")
        for i in range(1, len(self.locations)):
            ax.scatter(self.locations[i][0], self.locations[i][1], color="blue")
            ax.annotate(f"{self.demands[i]}", (self.locations[i][0] + 0.02, self.locations[i][1] + 0.02))

        for i in range(len(self.vehicle_history) - 1):
            ax.plot(
                [self.vehicle_history[i][0], self.vehicle_history[i + 1][0]],
                [self.vehicle_history[i][1], self.vehicle_history[i + 1][1]],
            )

        plt.title(f"Current capacity: {self.current_capacity}, Summed cost: {self.summed_cost: .2f}")
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.show()

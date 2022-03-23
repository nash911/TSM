import numpy as np
import time
from collections import OrderedDict

from utils import generate_adjacency_matrix


class DP(object):
    def __init__(self, X, path_mat, start_city=None):
        # Set the graph matrix X, by indicating a infinite cost for city pair {cᵢ, cⱼ} for which a
        # direct path of travel does not exist
        self._X = np.where(X >= 0, X, np.inf)

        # Total number of cities to be visited
        self._v = X.shape[0]

        # Create an adjacency matrix for the graph matrix X, as a boolean map, indicating if a
        # direct path of travel exists between each city pair {cᵢ, cⱼ}, ∀ i, j ∈ {1, 2, ..., |V|}.
        self._adj_mat = np.array(generate_adjacency_matrix(X), dtype=bool)

        # Remove self edges from the adjacency matrix
        np.fill_diagonal(self._adj_mat, False)

        # A boolean map as a 2D matrix, indicationg if a path of any length exists between each
        # city pairs {cᵢ, cⱼ}, ∀ i, j ∈ {1, 2, ..., |V|}.
        self._path_mat = path_mat

        # Remove self edges from the path matrix
        np.fill_diagonal(self._path_mat, False)

        # List of cities the journey can start from
        if start_city is None:
            self._start_cities = list(range(self._v))
        else:
            self._start_cities = [start_city - 1]

        # A counter for number of computations - For debugging purpose
        self._evaluation_counter = 0

        # A data structure mapping each city pair {cᵢ, cⱼ} to the shortest path needed to be
        # traversed between them.
        self._sub_paths = OrderedDict()
        for i in range(self._v):
            for j in range(self._v):
                if i == j:
                    # Self-edges marked as invalied
                    self._sub_paths[(i, j)] = None
                elif self._adj_mat[i, j]:
                    # The case were a direct path exists between cities cᵢ and cⱼ
                    self._sub_paths[(i, j)] = [j]
                elif self._path_mat[i, j]:
                    # The case were a direct path does not exists between cities cᵢ and cⱼ, but
                    # recheable indirectly. Set as an empty list to be filled in later.
                    self._sub_paths[(i, j)] = list()
                else:
                    # The case where city cⱼ cannot be reached from city cᵢ, and hence marked
                    # as invalid
                    self._sub_paths[(i, j)] = None

    def optimal_path(self, source, dest, end=None):
        if source == end:
            self._evaluation_counter += 1
            return 0, [end]
        elif len(dest) == 0:
            self._evaluation_counter += 1
            # Has reached the end of the tree (the final city), so returning 0 cost and empth path
            return 0, []
        elif len(dest) == 1:
            # Has reached the penultimate city
            dest = list(dest)[0]
            self._evaluation_counter += 1

            # Check if a path of any length exists between source city cᵢ and destination city cⱼ
            if self._path_mat[source, dest]:
                # If so, then the total cost = Travel_cost(cᵢ --> cⱼ) only
                # No accommodation at the final city
                return self._X[source, dest], [source] + self._sub_paths[(source, dest)]
            else:
                # If not, then the current path is invalid. Returning infinite cost and no path
                return np.inf, None
        else:
            costs = [np.inf]
            paths = [None]

            # Implementing TSM DP formula:
            # g(i, s) = min(w(i, j), g(j, {s - j})),
            #           j∈s
            # where i = source_city, j = destination_city, and s = set of cities yet to be visited.
            for j in dest:
                if end is not None:
                    # While finding the optimal path between two cities c_start and c_end, which
                    # are not directly connected (path retracing), check if source city cᵢ of the
                    # current iteration, and the destination city cⱼ are directly connected
                    if not self._adj_mat[source, j]:
                        # If not, then proceed no further through this branch
                        continue
                else:
                    # While finding the optimal path, starting from some city c_start (without a
                    # desired final destination), check if a path of any length exists between the
                    # source city cᵢ of the current iteration and the destination city cⱼ
                    if not self._path_mat[source, j]:
                        # If not, then proceed no further through this branch
                        continue

                # Get the minimum cost of travel and accommodation for the route starting from
                # city cⱼ and visiting the remaining cities in the cities-to-visit set.
                # Also, en route from the source city cᵢ to the destination city cⱼ, remove any
                # transit cities from the cities-to-visit set.
                cost, path = \
                    self.optimal_path(j, dest - set(self._sub_paths[(source, j)]), end)

                # If the cost of travel from the source city cᵢ to the destination city cⱼ is
                # infinite, then it indicates that the branch is not completely traversable
                if cost == np.inf:
                    continue
                else:
                    # Cost of travel from city cᵢ to all remaining cities via city cⱼ =
                    # Travel_cost(cᵢ --> cⱼ) + Hotel_cost(cⱼ) + Total future cost for travelling
                    # from cⱼ to the remaining cities via the optimal path
                    costs.append(self._X[source, j] + (0 if j == end else self._X[j, j]) + cost)

                    # Check if a direct path exists between the source city cᵢ and the destination
                    # city cⱼ
                    if self._adj_mat[source, j] or end is not None:
                        # If so, then append the source city cᵢ to the path list
                        paths.append([source] + path)
                    else:
                        # If not, then append both the source city cᵢ, and intermediate cities
                        # travelled en route to the destination city cⱼ
                        paths.append([source] + self._sub_paths[(source, j)] + path[1:])

            self._evaluation_counter += len(dest)

            # The min(∙) step of the DP formula
            #     j∈s
            cheapest_subrout = np.argmin(np.array(costs))
            return costs[cheapest_subrout], paths[cheapest_subrout]

    def solve(self, debugger=False):
        if debugger:
            start_time = time.time()

        # For each vertex pair {vᵢ, vⱼ}, check and create a flag if a path of length 1 < L < |V|
        # exists, ∀ i, j ∈ {1, 2, ..., |V|}, where L is the shortest path length between
        # vertices vᵢ and vⱼ
        retrace_flag = np.logical_xor(self._adj_mat, self._path_mat)

        # If there are any vertex pair {vᵢ, vⱼ}, with shortest path length 1 < L < |V|
        if np.any(retrace_flag):
            # Get all vertex pairs for which the above condition satisfies
            retrace_paths = np.array(np.where(np.array(retrace_flag, dtype=int) == 1)).T.tolist()

            # Retrace the optimal path (with the lowest travel + accommodation cost) between
            # vertex pair {vᵢ, vⱼ} s.t. 1 < L < |V|, ∀ i, j ∈ {1, 2, ..., |V|}.
            for i, j in retrace_paths:
                cost, path = self.optimal_path(i, set(range(self._v)) - {i}, j)

                # Set the optimal path and cost of travelling from vertex vᵢ to vⱼ
                self._sub_paths[(i, j)] = path[1:]
                self._X[i, j] = cost

        costs = list()
        paths = list()

        # Starting from each city cᵢ ∈ {starting_cities}, get the optimal path
        for s in self._start_cities:
            cost, path = self.optimal_path(s, set(range(self._v)) - {s})
            costs.append(cost)
            paths.append(path)

        # Among each city cᵢ ∈ {starting_cities}, get the most optimal city to start from
        cheapest_trip = np.argmin(np.array(costs))

        if debugger:
            analytical_eval_counts = \
                int(np.math.factorial(self._v) * (2 + np.sum([1.0 / np.math.factorial(i) for i in
                                                  list(range(2, self._v - 1))])))
            print("Number of computations: ", self._evaluation_counter)
            print("Analytical calculated for a connected graph: ", analytical_eval_counts)

            # Check if each city is visited atleast once
            np.testing.assert_array_equal(list(set(range(self._v))),
                                          list(set(paths[cheapest_trip])))

            end_time = time.time()
            print("Wall Time: %.2f(s)" % (end_time - start_time))
            print()

        try:
            return (np.array(paths[cheapest_trip]) + 1).tolist(), costs[cheapest_trip]
        except TypeError:
            return [], None

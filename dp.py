import numpy as np
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
        self._adj_mat = np.array(generate_adjacency_matrix(X), dtype=np.bool)

        # A boolean map as a 2D matrix, indicationg if a path of any length exists between each
        # city pairs {cᵢ, cⱼ}, ∀ i, j ∈ {1, 2, ..., |V|}.
        self._path_mat = path_mat

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

        # print("self._sub_paths: \n", self._sub_paths)
        # print("self._adj_mat:\n", self._adj_mat)

    def subpath_cost(self, start, end, cities_set):
        # print("[0] Start: ", (start + 1), "  End: ", (end + 1), "  Cities: ",
        #       set(np.array(list(cities_set)) + 1))
        if start == end:
            # print("[1] Start: ", (start + 1), "  End: ", (end + 1), "  Cities: ",
            #       set(np.array(list(cities_set)) + 1), "  Cost: ", 0, "  Path: ", [])
            return 0, [end]
        elif len(cities_set) == 1:
            # print("[2] Start: ", (start + 1), "  End: ", (end + 1), "  Cities: ",
            #       set(np.array(list(cities_set)) + 1), "  Cost: ", self._X[start, list(cities_set)[0]],
            #       "  Path: ", (np.array([start] + list(cities_set)) + 1).tolist())
            return self._X[start, list(cities_set)[0]], [start] + list(cities_set)
        else:
            costs = list()
            paths = list()

            for i in cities_set:
                if not self._adj_mat[start, i]:
                    # print("(%d, %d): None" % (start + 1, i + 1))
                    # print("[3] Start: ", (start + 1), "  End: ", (end + 1), "  Cities: ",
                    #       set(np.array(list(cities_set)) + 1), "  Cost: ", np.inf, "  Path: None")
                    # return np.inf, None
                    costs.append(np.inf)
                    paths.append(None)
                else:
                    # sub_dest = copy.deepcopy(dest)
                    # sub_dest.remove(d)
                    travel_cost, path = self.subpath_cost(i, end, cities_set - {i})
                    if travel_cost == np.inf:
                        costs.append(np.inf)
                        paths.append([])
                    else:
                        # costs.append(self._X[source, d] + travel_cost)  # Travel cost only
                        # Travel cost + Hotel cost
                        costs.append(self._X[start, i] + (0 if i == end else self._X[i, i]) +
                                     travel_cost)
                        paths.append([start] + path)
                        # paths.append(path)

            cheapest_subrout = np.argmin(np.array(costs))
            # print("[4] Start: ", (start + 1), "  End: ", (end + 1), "  Cities: ",
            #       set(np.array(list(cities_set)) + 1), "  Cost: ", costs[cheapest_subrout],
            #       "  Path: ", (np.array(paths[cheapest_subrout]) + 1).tolist())
            return costs[cheapest_subrout], paths[cheapest_subrout]

    def retrace_path(self, start, end):
        cities_set = set(range(self._v))

        cost, path = self.subpath_cost(start, end, cities_set - {start})

        self._X[start, end] = cost
        self._sub_paths[(start, end)] = path[1:]

    def cost(self, source, dest):
        if len(dest) == 0:
            # Has reached the end of the tree (the final city)
            return 0, []
        elif len(dest) == 1:
            # Has reached the penultimate city
            dest = list(dest)[0]
            self._evaluation_counter += 1

            # Check if a path or any length exists between source city cᵢ and destination city cⱼ
            if self._path_mat[source, dest]:
                # If so, then the total cost = Travel_cost(cᵢ --> cⱼ) only
                # No accommodation at the final city
                return self._X[source, dest], [source] + self._sub_paths[(source, dest)]
            else:
                # If not, then the current path is invalid. Returning infinite cost and no path
                return np.inf, None
        else:
            costs = list()
            paths = list()

            # Implementing TSM DP formula:
            # g(i, s) = min(w(i, j), g(j, {s - j})),
            #           j∈s
            # where i = source_city, j = destination_city, and s = set of cities yet to be visited.
            for j in dest:
                # Check if a path or any length exists between the source city cᵢ and the
                # destination city cⱼ
                if not self._path_mat[source, j]:
                    # If not, then proceed no further through that branch, and mark cost of travel
                    # as infinite
                    costs.append(np.inf)
                    paths.append(None)
                    continue

                # Get the minimum cost of travel and accommodation for the route starting from
                # city cⱼ and visiting the remaining cities in the cities-to-visit set.
                # Also, en route from the source city cᵢ to the destination city cⱼ, remove any
                # transit cities from the cities-to-visit set.
                travel_cost, path = self.cost(j, dest - set(self._sub_paths[(source, j)]))

                # If the cost of travel from the source city cᵢ to the destination city cⱼ is
                # infinite, then it indicates that the branch is not completely traversable
                if travel_cost == np.inf:
                    costs.append(np.inf)
                    paths.append(None)
                else:
                    # Cost of travel from city cᵢ to cⱼ = Travel_cost(cᵢ --> cⱼ) + Hotel_cost(cⱼ)
                    costs.append(self._X[source, j] + self._X[j, j] + travel_cost)

                    # Check if a direct path exists between the source city cᵢ and the destination
                    # city cⱼ
                    if self._adj_mat[source, j]:
                        # If so, the append the source city cᵢ to the path list
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

    def solve(self):
        # For each vertex pair {vᵢ, vⱼ}, check and create a flag if a path of length 1 < L < |V|
        # exists, ∀ i, j ∈ {1, 2, ..., |V|}, where L is the shortest path length between
        # vertices vᵢ and vⱼ
        retrace_flag = np.logical_xor(self._adj_mat, self._path_mat)

        # If there are any vertex pair {vᵢ, vⱼ}, with shortest path length 1 < L < |V|
        if np.any(retrace_flag):
            # Get all vertex pairs for which the above condition satisfies
            retrace_paths = np.array(np.where(np.array(retrace_flag, dtype=np.int) == 1)).T.tolist()

            # Retrace the optimal path (with the lowest travel + accommodation cost) between
            # vertex pair {vᵢ, vⱼ} s.t. 1 < L < |V|, ∀ i, j ∈ {1, 2, ..., |V|}.
            for rp in retrace_paths:
                self.retrace_path(rp[0], rp[1])
                # print("(%d --> %d)" % (rp[0] + 1, rp[1] + 1))

        # print("\nself._sub_paths: \n", self._sub_paths)
        # print("self._X:\n", self._X)

        costs = list()
        paths = list()
        to_visit = set(range(self._v))

        # Starting from each city cᵢ ∈ {starting_cities}, get the optimal path
        for s in self._start_cities:
            cost, path = self.cost(s, to_visit - {s})
            costs.append(cost)
            paths.append(path)

        # Among each city cᵢ ∈ {starting_cities}, get the most optimal city to start from
        cheapest_trip = np.argmin(np.array(costs))

        # print("Paths: ", (np.array(paths) + 1).tolist())
        print("Counts: ", self._evaluation_counter)

        analytical_eval_counts = \
            int(np.math.factorial(self._v) * (2 + np.sum([1.0 / np.math.factorial(i) for i in
                                              list(range(2, self._v - 1))])))

        print("Analytical calculated evaluation counts: ", analytical_eval_counts)

        return (np.array(paths[cheapest_trip]) + 1).tolist(), costs[cheapest_trip]

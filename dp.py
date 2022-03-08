import numpy as np
import copy

from utils import generate_adjacency_matrix


class DP(object):
    def __init__(self, X, start_city=None):
        self._X = X
        self._v = X.shape[0]
        self._adj_mat = generate_adjacency_matrix(X)
        if start_city is None:
            self._start_cities = list(range(self._v))
        else:
            self._start_cities = [start_city - 1]
        self._evaluation_counter = 0

    def cost(self, source, dest):
        if len(dest) == 1:
            # print("source: ", (source + 1), "  destinations: ", (np.array(dest) + 1).tolist(),
            #       "  Path: ", (np.array(dest) + 1).tolist(), "  Cost: ", self._X[source, dest[0]])
            self._evaluation_counter += 1
            return self._X[source, dest[0]], [source] + dest
        else:
            costs = list()
            paths = list()

            for d in dest:
                sub_dest = copy.deepcopy(dest)
                sub_dest.remove(d)
                travel_cost, path = self.cost(d, sub_dest)
                # costs.append(self._X[source, d] + travel_cost)  # Travel cost only
                # Travel cost + Hotel cost
                costs.append(self._X[source, d] + self._X[d, d] + travel_cost)
                paths.append([source] + path)
            cheapest_subrout = np.argmin(np.array(costs))
            # print("Source: ", (source + 1), "  Destinations: ", (np.array(dest) + 1).tolist(),
            #       "  Path: ", (np.array(paths[cheapest_subrout]) + 1).tolist(),
            #       "  Cost: ", costs[cheapest_subrout])
            self._evaluation_counter += len(dest)
            return costs[cheapest_subrout], paths[cheapest_subrout]

    def solve(self):
        costs = list()
        paths = list()
        for s in self._start_cities:
            to_visit = list(range(self._v))
            to_visit.remove(s)
            cost, path = self.cost(s, to_visit)
            costs.append(cost)
            paths.append(path)

        cheapest_trip = np.argmin(np.array(costs))

        # print("Paths: ", (np.array(paths) + 1).tolist())
        # print("Counts: ", self._evaluation_counter)

        analytical_eval_counts = \
            int(np.math.factorial(self._v) * (2 + np.sum([1.0 / np.math.factorial(i) for i in
                                              list(range(2, self._v - 1))])))

        # print("Analytical calculated evaluation counts: ", analytical_eval_counts)
        #
        # print("Min Trip Cost: ", costs[cheapest_trip])
        # print("Trip Path:")
        # path_string = ''
        # for p in paths[cheapest_trip]:
        #     path_string += "%d --> "
        # print(path_string % tuple(np.array(paths[cheapest_trip]) + 1))

        return (np.array(paths[cheapest_trip]) + 1).tolist(), costs[cheapest_trip]

import numpy as np
import copy
from collections import OrderedDict

from utils import generate_adjacency_matrix


class DP(object):
    def __init__(self, X, path_mat, start_city=None):
        self._X = np.where(X >= 0, X, np.inf)
        print("X:\n", X)
        print("self._X:\n", self._X)
        self._v = X.shape[0]
        self._path_mat = path_mat
        self._adj_mat = np.array(generate_adjacency_matrix(X), dtype=np.bool)
        if start_city is None:
            self._start_cities = list(range(self._v))
        else:
            self._start_cities = [start_city - 1]
        self._evaluation_counter = 0

        self._sub_paths = OrderedDict()
        for i in range(self._v):
            for j in range(self._v):
                if i == j:
                    self._sub_paths[(i, j)] = None
                elif self._adj_mat[i, j]:
                    self._sub_paths[(i, j)] = [j]
                elif self._path_mat[i, j]:
                    self._sub_paths[(i, j)] = list()
                else:
                    self._sub_paths[(i, j)] = None

        print("self._sub_paths: \n", self._sub_paths)
        print("self._adj_mat:\n", self._adj_mat)

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
        # cities_set = list(range(self._v))
        # cities_set.remove(start)

        cities_set = set(range(self._v))
        # cities_set.remove(start)

        cost, path = self.subpath_cost(start, end, cities_set - {start})

        self._X[start, end] = cost
        self._sub_paths[(start, end)] = path[1:]
        print("Sub_path(", (start + 1), "-->", (end + 1), "): ", (np.array(path) + 1).tolist())

        # costs.append(cost)
        # paths.append(path)
        # cheapest_trip = np.argmin(np.array(costs))

    def cost(self, source, dest):
        if len(dest) == 1:
            self._evaluation_counter += 1
            # print("[1] Source: ", source, "  Destination: ", dest[0])
            # if not (self._adj_mat[source, dest[0]] or self._path_mat[source, dest[0]]):
            if not self._path_mat[source, dest[0]]:
                return np.inf, None
            else:
                if source == 3 and dest[0] == 4:
                    print("[1] Source: ", source + 1, "  Destination: ", dest[0] + 1, "  Cost: ",
                          self._X[source, dest[0]], "  Path: ",
                          (np.array([source] + self._sub_paths[(source, dest[0])]) + 1).tolist())
                return self._X[source, dest[0]], [source] + self._sub_paths[(source, dest[0])] #dest
        else:
            costs = list()
            paths = list()

            # print("[2] Source: ", source + 1, "  dest: ", (np.array(dest) + 1).tolist())

            for d in dest:
                # print("(s: %d, d: %d)" % (source + 1, d + 1))
                if not self._path_mat[source, d]:
                    costs.append(np.inf)
                    paths.append(None)
                    continue
                sub_dest = copy.deepcopy(dest)
                sub_dest.remove(d)
                # print("(d: ", (d + 1), "  sub_dest:", (np.array(sub_dest) + 1).tolist())
                travel_cost, path = self.cost(d, sub_dest)
                if travel_cost == np.inf:
                    costs.append(np.inf)
                    paths.append(None)
                else:
                    # Travel cost + Hotel cost
                    costs.append(self._X[source, d] + self._X[d, d] + travel_cost)
                    if source == 4 and d == 1:
                        print("[3] Source: ", source + 1, "  D: ", d + 1, "  Sub_dest: ",
                              (np.array(sub_dest) + 1).tolist(), "  Cost: ", travel_cost,
                              "  Path: ", (np.array(path) + 1).tolist(), "  Sub_path: ",
                              (np.array(self._sub_paths[(source, d)]) + 1).tolist())
                    # paths.append([source] + path)
                    # paths.append(self._sub_paths[(source, d)] + path)
                    if self._adj_mat[source, d]:
                        paths.append([source] + path)
                    else:
                        paths.append([source] + self._sub_paths[(source, d)] + path[1:])
            cheapest_subrout = np.argmin(np.array(costs))
            # print("[4] Source: ", (source + 1), "  Destinations: ", (np.array(dest) + 1).tolist(),
            #       "  Path: ", (np.array(paths[cheapest_subrout])).tolist(),
            #       "  Cost: ", costs[cheapest_subrout])
            self._evaluation_counter += len(dest)
            return costs[cheapest_subrout], paths[cheapest_subrout]

    # def cost(self, source, dest):
    #     if len(dest) == 1:
    #         # print("source: ", (source + 1), "  destinations: ", (np.array(dest) + 1).tolist(),
    #         #       "  Path: ", (np.array(dest) + 1).tolist(), "  Cost: ", self._X[source, dest[0]])
    #         self._evaluation_counter += 1
    #         return self._X[source, dest[0]], [source] + dest
    #     else:
    #         costs = list()
    #         paths = list()
    #
    #         for d in dest:
    #             sub_dest = copy.deepcopy(dest)
    #             sub_dest.remove(d)
    #             travel_cost, path = self.cost(d, sub_dest)
    #             # costs.append(self._X[source, d] + travel_cost)  # Travel cost only
    #             # Travel cost + Hotel cost
    #             costs.append(self._X[source, d] + self._X[d, d] + travel_cost)
    #             paths.append([source] + path)
    #         cheapest_subrout = np.argmin(np.array(costs))
    #         # print("Source: ", (source + 1), "  Destinations: ", (np.array(dest) + 1).tolist(),
    #         #       "  Path: ", (np.array(paths[cheapest_subrout]) + 1).tolist(),
    #         #       "  Cost: ", costs[cheapest_subrout])
    #         self._evaluation_counter += len(dest)
    #         return costs[cheapest_subrout], paths[cheapest_subrout]

    def solve(self):
        paths_bool = np.logical_xor(self._adj_mat, self._path_mat)
        if np.any(paths_bool):
            retrace_paths = np.array(np.where(np.array(paths_bool, dtype=np.int) == 1)).T.tolist()
            print("Retrace Paths of: ", retrace_paths) #, (np.array(retrace_paths) + 1).T.tolist())

            for rp in retrace_paths:
                self.retrace_path(rp[0], rp[1])
                print("(%d --> %d)" % (rp[0] + 1, rp[1] + 1))

        print("\nself._sub_paths: \n", self._sub_paths)
        print("self._X:\n", self._X)

        costs = list()
        paths = list()
        for s in self._start_cities:
            to_visit = list(range(self._v))
            to_visit.remove(s)
            cost, path = self.cost(s, to_visit)
            costs.append(cost)
            paths.append(path)
            # print("s: ", s, "  Costs: ", costs, "  Path: ", paths)

        cheapest_trip = np.argmin(np.array(costs))

        # print("Paths: ", (np.array(paths) + 1).tolist())
        print("Counts: ", self._evaluation_counter)

        analytical_eval_counts = \
            int(np.math.factorial(self._v) * (2 + np.sum([1.0 / np.math.factorial(i) for i in
                                              list(range(2, self._v - 1))])))

        print("Analytical calculated evaluation counts: ", analytical_eval_counts)

        # print("Min Trip Cost: ", costs[cheapest_trip])
        # print("Trip Path:")
        # path_string = ''
        # for p in paths[cheapest_trip]:
        #     path_string += "%d --> "
        # print(path_string % tuple(np.array(paths[cheapest_trip]) + 1))

        return (np.array(paths[cheapest_trip]) + 1).tolist(), costs[cheapest_trip]

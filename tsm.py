import numpy as np
import sys
import os
import getopt

from dp import DP
from utils import generate_adjacency_matrix
from utils import extract_graph_data
from utils import plot_graph_network


def test_solutions(X, v, path_mat, file, start_city, optimal_path, optimal_cost):
    # Plot the graph for visualization
    plot_graph_network(X, v, 2)

    path, cost = solve_tsm_problem(X, path_mat, start_city)

    print("Path: ", ','.join(str(p) for p in path))
    print("Cost:", cost)

    try:
        np.testing.assert_array_equal(optimal_path, path)
        np.testing.assert_array_equal([optimal_cost], [cost])
    except AssertionError:
        print("AssertionError: File: %s - start_city: %s" % (file, str(start_city)))
        sys.exit(1)

    return


def validate_graph(X, v):
    # Create an adjacency matrix of graph matrix X
    adj_mat = generate_adjacency_matrix(X)

    # Remove self edges from the adjacency matric for convenience
    np.fill_diagonal(adj_mat, 0)

    # # Calculate total number of undirected edges in the graph
    # e = (np.sum(np.array((adj_mat + adj_mat.T) > 0, dtype=np.int))) / 2.0
    # if int(e) != e:
    #     print("Error: ", int(e), " != ", e)
    #     sys.exit(2)
    # else:
    #     e = int(e)
    # print("Number of edges: ", e)

    # Compute number of unique paths from vertex vᵢ to vⱼ with exactly n hops, where
    # n ∈ {1, 2, ... v-1}
    adj_powers = [adj_mat]
    for i in range(2, v):
        # No. of unique paths from vertex vᵢ to vⱼ with exactly n hops = (Adj)ⁿ
        adj_powers.append(np.linalg.matrix_power(adj_mat, i))

    # Summing all possible paths from vertex vᵢ to vⱼ with n hops, where
    # n ∈ {1, 2, ... v-1}
    paths_sum = np.abs(np.sum(np.array(adj_powers), axis=0))

    # Check if atlest one path exists between vertices vᵢ and vⱼ with any hop of length n,
    # ∀ i, j ∈ {1, 2, ... v}, and ∀ n ∈ {1, 2, ... v-1}
    path_bool = np.array(paths_sum, dtype=np.bool)
    np.fill_diagonal(path_bool, True)
    path_exists = np.logical_or(path_bool, path_bool.T)

    # If there does not exists a path between any vertex pair {vᵢ, vⱼ}, then the graph is
    # not traversable
    if not np.all(path_exists):
        return False, [], np.empty(0)
    else:
        zero_cols = np.array(np.abs(np.sum(paths_sum, axis=0)) > 0, dtype=np.int)
        zero_rows = np.array(np.abs(np.sum(paths_sum, axis=1)) > 0, dtype=np.int)

        if np.sum(zero_cols) < v:
            ind = np.argwhere(zero_cols == 0).flatten()
            return True, [ind[0] + 1], path_bool
        elif np.sum(zero_rows) < v:
            ind = np.argwhere(zero_rows == 0).flatten()
            possible_cities = list(range(1, (v + 1)))
            possible_cities.remove(ind[0] + 1)
            return True, possible_cities, path_bool

    return True, list(range(1, (v + 1))), path_bool


def solve_tsm_problem(X, path_mat, start_city=None):
    dp_solver = DP(X, path_mat, start_city)
    return dp_solver.solve()


def usage():
    print("Usage: tsm.py [-h | --help] \n"
          "              [-i | --input] <Path to input file/dir containing graph data> \n"
          "              [-s | --start_city] <Index of city to start the journey from> \n")


def main(argv):
    input = None
    start_city = None
    force_eval = False

    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["help", "input=", "start_city="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(1)
        elif opt in ("-i", "--input"):
            if not os.path.exists(arg):
                print("Error: Invalid input path: %s. Please provide a valid file/dir path." % arg)
                sys.exit(1)
            else:
                input = arg
        elif opt in ("-s", "--start_city"):
            try:
                if int(arg) < 1:
                    print("Error: The start-city index should be in the range: [1, No. of cities].")
                    sys.exit(1)
                else:
                    start_city = int(arg)
            except ValueError:
                print("Error: The start-city index should be an integer value in the range: " +
                      "[1, No. of cities].")
                sys.exit(1)

    graph_dict = extract_graph_data(input)

    for k, val in graph_dict.items():
        X = val['graph_mat']
        v = val['v']
        graph_valid, valid_start_cities, path_mat = validate_graph(X, v)

        if not graph_valid:
            print("Nonviable input")
            return

        if val['solutions'] is None or force_eval:
            if start_city is not None and start_city not in valid_start_cities:
                print(("It is not possible to visit all the cities by starting the journey from " +
                      "city %d") % v)
                print("For the provided graph, valid starting city/cities is/are: ",
                      valid_start_cities)
                return

            # Plot the graph for visualization
            plot_graph_network(X, v)

            path, trip_cost = solve_tsm_problem(X, path_mat, start_city)

            print(','.join(str(p) for p in path), "\n")
            print(trip_cost)
        else:
            for sol in val['solutions']:
                test_solutions(X, v, path_mat, k, start_city=sol[0], optimal_path=sol[1],
                               optimal_cost=sol[2])

    return


if __name__ == "__main__":
    main(sys.argv[1:])

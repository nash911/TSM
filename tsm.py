import numpy as np
import sys
import os
import getopt

from dp import DP
from utils import generate_adjacency_matrix
from utils import extract_graph_data
from utils import plot_graph_network


def test_solutions(X, v, path_mat, file, start_city, optimal_path, optimal_cost, args=None,
                   debugger=False):
    if debugger:
        # Plot the graph for visualization
        plot_graph_network(X, v, t=2)

    if args is not None:
        # To optimize for the total number of days of travel, instead of total cost
        X = np.where(X >= 0, 1, X)
        np.fill_diagonal(X, 0)

    path, cost = solve_tsm_problem(X, path_mat, start_city, debugger)

    if debugger:
        print("Optimal Path: ", ','.join(str(p) for p in path))
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

    # Remove self edges from the adjacency matrix for convenience
    np.fill_diagonal(adj_mat, 0)

    # Finding the total number of paths from vertex vᵢ to vⱼ, ∀ i, j ∈ {1, 2, ... v}:
    # Compute the number of unique paths from vertex vᵢ to vⱼ with exactly n hops, and sum over
    # all n ∈ {1, 2, ..., |V|-1}
    paths_sum = np.array(np.copy(adj_mat), dtype=float)  # n=1

    # Using the eigen trick to find the nth power of a matrix as follows:
    # Adj = PDP⁻¹, where D is a diagonal matrix
    # (Adj)ⁿ = (PDP⁻¹)ⁿ = PDⁿP⁻¹
    eigen_val, P = np.linalg.eig(adj_mat)
    try:
        P_inv = np.linalg.inv(P)
        for i in range(2, v):
            # No. of unique paths from vertex vᵢ to vⱼ with exactly n hops = (Adj)ⁿ
            D = np.diag(eigen_val**i)
            paths_sum += np.matmul(np.matmul(P, D), P_inv).real
    except np.linalg.linalg.LinAlgError:
        # If P is a singular matrix (when any vertex vᵢ has either no incoming edges,
        # or no outgoing edges), then it is noninvertable.
        # So, resorting to regular matrix multiplication for finding the nth power of the
        # adjacency matrix.
        for i in range(2, v):
            # No. of unique paths from vertex vᵢ to vⱼ with exactly n hops = (Adj)ⁿ
            paths_sum += np.linalg.matrix_power(adj_mat, i)

    # Check if atlest one path exists between vertices vᵢ and vⱼ with any hop of length n,
    # ∀ i, j ∈ {1, 2, ..., |V|}, and ∀ n ∈ {1, 2, ..., |V|-1}
    path_bool = np.array(np.rint(np.abs(paths_sum)), dtype=bool)
    np.fill_diagonal(path_bool, True)
    path_exists = np.logical_or(path_bool, path_bool.T)

    # If there does not exists a path between any vertex pair {vᵢ, vⱼ}, then the graph is
    # not traversable
    if not np.all(path_exists):
        return False, [], np.empty(0)
    else:
        # Find the list of possible cities from where the journey can originate
        zero_cols = np.array(np.abs(np.sum(paths_sum, axis=0)) > 0, dtype=int)
        zero_rows = np.array(np.abs(np.sum(paths_sum, axis=1)) > 0, dtype=int)

        if np.sum(zero_cols) < v:
            # In the path_matrix, if there exists a column Cᵢ = [0, 0, ..., 0]ᵀ, then vertex vᵢ
            # has no incoming edges, so the journey can only start from vertex vᵢ
            ind = np.argwhere(zero_cols == 0).flatten()
            return True, [ind[0] + 1], path_bool
        elif np.sum(zero_rows) < v:
            # In the path_matrix, if there exists a row Rᵢ = [0, 0, ..., 0], then vertex vᵢ
            # has no outgoing edges, so the journey can start from any vertex vⱼ, where j≠i
            ind = np.argwhere(zero_rows == 0).flatten()
            possible_cities = list(range(1, (v + 1)))
            possible_cities.remove(ind[0] + 1)
            return True, possible_cities, path_bool

    return True, list(range(1, (v + 1))), path_bool


def solve_tsm_problem(X, path_mat, start_city=None, debugger=False):
    dp_solver = DP(X, path_mat, start_city)
    return dp_solver.solve(debugger)


def usage():
    print("Usage: tsm.py [-h | --help] \n"
          "              [-d | --debugger] <Debugger flag to display network graph and verbose" +
          "output> \n"
          "              [-t | --time] <Flag to optimize for time of travel instead of cost> \n"
          "              [-i | --input] <Path to input file/dir containing graph data> \n"
          "              [-s | --start_city] <Index of city to start the journey from> \n")


def main(argv):
    input = None
    start_city = None
    time = False
    debugger = False

    try:
        opts, args = getopt.getopt(argv, "hdti:s:", ["help", "debugger", "time", "input=",
                                                     "start_city="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            return
        elif opt in ("-d", "--debugger"):
            debugger = True
        elif opt in ("-t", "--time"):
            time = True
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

        if time:
            # To optimize for the total number of days of travel, instead of total cost
            X = np.where(X >= 0, 1, X)
            np.fill_diagonal(X, 0)

        graph_valid, valid_start_cities, path_mat = validate_graph(X, v)

        if not graph_valid:
            print("Nonviable input")
            return

        if val['solutions'] is None:
            if start_city is not None and start_city not in valid_start_cities:
                print(("It is not possible to visit all the cities by starting the journey from " +
                      "city %d") % start_city)
                print("For the provided graph, valid starting city/cities is/are: ",
                      valid_start_cities)
                return

            if debugger:
                # Plot the graph for visualization
                plot_graph_network(X, v, t=10)

            path, trip_cost = solve_tsm_problem(X, path_mat, start_city, debugger)

            if len(path) == 0 and trip_cost is None:
                print(("It is not possible to visit all the cities by starting the journey from " +
                      "city %d") % start_city)
            else:
                print(','.join(str(p) for p in path), "\n")
                print(trip_cost)
        else:
            for sol in val['solutions']:
                test_solutions(X, v, path_mat, k, start_city=sol[0], optimal_path=sol[1],
                               optimal_cost=sol[2], args=sol[3], debugger=debugger)

    return


if __name__ == "__main__":
    main(sys.argv[1:])

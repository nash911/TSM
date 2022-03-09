import numpy as np
import sys
import os
import getopt
import csv
import matplotlib.pyplot as plt
import networkx as nx

from dp import DP
from utils import generate_adjacency_matrix


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


def plot_graph_network(X, v):
    # Create an adjacency matrix of graph matrix X
    adj_mat = generate_adjacency_matrix(X)

    # Create a list of directed edges associated with vertices vᵢ and vⱼ, ∀ i, j ∈ {1, 2, ... v}
    graph_edges = list()
    for i, row in enumerate(adj_mat):
        for j, val in enumerate(row):
            if val == 1:
                graph_edges.append((str(i + 1), str(j + 1)))

    # Create a directed graph, with the associated edges
    G = nx.DiGraph()
    G.add_edges_from(graph_edges)

    # Plot the graph for visualization
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_size=1000, connectionstyle='arc3, rad = 0.5')
    plt.waitforbuttonpress(0)
    plt.close()
    return


def solve_tsm_problem(X, path_mat, start_city=None):
    dp_solver = DP(X, path_mat, start_city)

    return dp_solver.solve()


def usage():
    print("Usage: tsm.py [-h | --help] \n"
          "              [-i | --inp_file] <Path to inpit file containing graph data> \n"
          "              [-s | --start_city] <Index of city to start the journey from> \n")


def main(argv):
    input_file = None
    start_city = None

    known_optimal_cost = None
    known_optimal_path = None

    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["help", "inp_file=", "start_city="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(1)
        elif opt in ("-i", "--inp_file"):
            if not os.path.isfile(arg):
                print("Error: Invalid input file path: %s. Please provide a valid file path." % arg)
                sys.exit(1)
            else:
                input_file = arg
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

    if input_file is None:
        print("Please provide a valid input file path, containing graph data.")
        print("Use flag [-i|--inp_file]")
        sys.exit(1)
    else:
        with open(input_file) as inp_f:
            csv_reader = csv.reader(inp_f, delimiter=',')
            graph_data = list()
            for i, row in enumerate(csv_reader):
                if i == 0:
                    v = int(row[0])
                elif i == 1:
                    # Omit the blank line
                    pass
                else:
                    # Extract graph rows from input file
                    graph_data.append(row)

    X = np.array(graph_data[:v], dtype=np.float)

    if len(graph_data) > v:
        known_optimal_cost = graph_data[-1]
        known_optimal_path = graph_data[-3]

    if X.shape[0] != X.shape[1]:
        print("Error: The provided graph matrix is not square!")
        sys.exit(1)
    elif X.shape[0] != v:
        print("Error: No. of cities in the input file is not consistent with the graph matrix size")
        sys.exit(1)
    elif start_city is not None and start_city > v:
        print("Error: The start-city index should be in the range: [1, %d]." % v)
        sys.exit(1)

    graph_valid, valid_start_cities, path_mat = validate_graph(X, v)

    if not graph_valid:
        print("Nonviable input")
        return
    elif start_city is not None and start_city not in valid_start_cities:
        print("It is not possible to visit all the cities by starting the journey from city %d" % v)
        print("For the provided graph, valid starting city/cities is/are: ", valid_start_cities)
        return

    # Plot the graph for visualization
    plot_graph_network(X, v)

    optimal_path, trip_cost = solve_tsm_problem(X, path_mat, start_city)

    print(','.join(str(p) for p in optimal_path))
    print("\n", trip_cost)

    return


if __name__ == "__main__":
    main(sys.argv[1:])

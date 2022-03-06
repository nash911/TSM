import numpy as np
import sys
import getopt
import csv
import matplotlib.pyplot as plt
import networkx as nx


def generate_random_graph(v, p_t, p_h):
    # A random matrix of size [v x v]
    X = np.random.uniform(-1, 1, (v, v))

    # Make diagonals (hotel cost) strictly positive
    np.fill_diagonal(X, np.abs(np.diag(X)))
    # free_mask = np.array((np.random.choice([0, 1], size=(n, n), p=[p, (1 - p)]) + np.eye(n)) > 0,
    #                      dtype=np.int)

    # Randomly choose some travels to be free with probability p_t
    free_mask = np.random.choice([0, 1], size=(v, v), p=[p_t, (1 - p_t)])

    # Replace diagonals of the free_mask to make randomly chosen hotel costs free with prob. p_h
    np.fill_diagonal(free_mask, np.random.choice([0, 1], size=(v), p=[p_h, (1 - p_h)]))

    # Apply free travel/accommadation mask and return
    return X * free_mask


def valid_graph(X, v):
    # Create an adjacency matrix of graph matrix X
    adj_mat = np.array(X >= 0, dtype=np.int)

    # Remove self edges from the adjacency matric for convenience
    np.fill_diagonal(adj_mat, 0)

    print("X:\n", X)
    print("adj_mat:\n", adj_mat)

    # Calculate total number of undirected edges in the graph
    e = (np.sum(np.array((adj_mat + adj_mat.T) > 0, dtype=np.int))) / 2.0
    if int(e) != e:
        print("Error: ", int(e), " != ", e)
        sys.exit(2)
    else:
        e = int(e)
    print("Number of edges: ", e)

    # Compute number of unique paths from vertex vᵢ to vⱼ with exactly n hops, where
    # n ∈ {1, 2, ... v-1}
    adj_powers = [adj_mat]
    for i in range(2, v):
        # No. of unique paths from vertex vᵢ to vⱼ with exactly n hops = (Adj)ⁿ
        adj_powers.append(np.linalg.matrix_power(adj_mat, i))

    # Summing all possible paths from vertex vᵢ to vⱼ with n hops, where
    # n ∈ {1, 2, ... v-1}
    paths_sum = np.abs(np.sum(np.array(adj_powers), axis=0))

    # Check if atlest one path exists between vertices vᵢ and vⱼ with n hops,
    # ∀ i, j ∈ {1, 2, ... v}, and ∀ n ∈ {1, 2, ... v-1}
    path_bool = np.array(paths_sum, dtype=np.bool)
    np.fill_diagonal(path_bool, True)
    path_exists = np.logical_or(path_bool, path_bool.T)

    print("Sum of Adj. powers:\n", np.sum(np.array(adj_powers), axis=0))

    # If there does not exists a path between any vertex pair {vᵢ, vⱼ}, then the graph is
    # not traversable
    if not np.all(path_exists):
        return False
    else:
        zero_cols = np.array(np.abs(np.sum(paths_sum, axis=0)) > 0, dtype=np.int)
        zero_rows = np.array(np.abs(np.sum(paths_sum, axis=1)) > 0, dtype=np.int)

        print("Path Zero Rows:", zero_rows)
        print("Path Zero Cols:", zero_cols)

        if np.sum(zero_cols) < v:
            ind = np.argwhere(zero_cols == 0).flatten()
            if ind.size != 1:
                print("Error in Zero_Cols: ", zero_cols)
                sys.exit(1)
            print("The journey can only start from city: %d" % (ind[0] + 1))
            return True
        elif np.sum(zero_rows) < v:
            ind = np.argwhere(zero_rows == 0).flatten()
            if ind.size != 1:
                print("Error in Zero_Rows: ", zero_rows)
                sys.exit(1)
            print("The journey can start from any city but: %d" % (ind[0] + 1))
            return True

    print("The journey can start from any city.")
    return True


def plot_graph_network(X, v):
    # Create an adjacency matrix of graph matrix X
    adj_mat = np.array(X >= 0, dtype=np.int)

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


def usage():
    print("Usage: data_generator.py [-h | --help] \n"
          "                         [-i | --invalid] <flag to create an invalid graph> \n"
          "                         [-n | --num_cities] <Number of cities to visit> \n"
          "                         [-H | --hotel_prob] <Free hotel prob> \n"
          "                         [-T | --travel_prob] <Free travel prob> \n")


def main(argv):
    invalid = False
    v = 3
    p_t = 0.1
    p_h = 0.1

    try:
        opts, args = getopt.getopt(argv, "hin:H:T:", ["help", "invalid", "num_cities=",
                                                      "hotel_prob=", "travel_prob="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-i", "--invalid"):
            invalid = True
        elif opt in ("-n", "--num_cities"):
            v = int(arg)
        elif opt in ("-H", "--hotel_prob"):
            if float(arg) < 0 or float(arg) > 1:
                print("Error: Free hotel prob should be 0 >= [-H|--hotel_prob] <= 1.0")
                sys.exit()
            else:
                p_h = float(arg)
        elif opt in ("-T", "--travel_prob"):
            if float(arg) < 0 or float(arg) > 1:
                print("Error: Free travel prob should be 0 >= [-T|--travel_prob] <= 1.0")
                sys.exit()
            else:
                p_t = float(arg)

    # Generate a random directed graph with v vertices, represented as matrix X
    X = generate_random_graph(v, p_t, p_h)

    # Validate the graph to check if th graph is fully traversable
    if invalid:
        while valid_graph(X, v):
            X = generate_random_graph(v, p_t, p_h)
    else:
        while not valid_graph(X, v):
            X = generate_random_graph(v, p_t, p_h)

    # Save the graph data in CSV format
    with open('inputs/graph_file.dat', 'w') as f:
        write = csv.writer(f)
        write.writerow(str(v))
        write.writerow('')
        for row in X:
            write.writerow(row.tolist())

    # Plot the graph for visualization
    plot_graph_network(X, v)

    return


if __name__ == "__main__":
    main(sys.argv[1:])

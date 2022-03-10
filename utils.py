import numpy as np
import sys
from os import listdir
from os.path import isdir, isfile, join
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
import networkx as nx


def generate_adjacency_matrix(X):
    # Adjacency matrix of graph matrix X
    return np.array(X >= 0, dtype=np.int)


def extract_graph_data(input_path, start_city=None):
    if input_path is None:
        print("Please provide a valid input file/dir path, containing graph data.")
        print("Use flag [-i|--input]")
        sys.exit(1)
    elif isfile(input_path):
        input_files = [input_path]
    elif isdir(input_path):
        # If the provided input path is a directory, then extract files in the directory
        input_files = [join(input_path, f) for f in listdir(input_path) if
                       isfile(join(input_path, f))]
    else:
        print("Error: Invalid input path: %s. Please provide a valid file/dir path." % input_path)
        sys.exit(1)

    # Create a directory with each item associated with a single input file, containing graph data,
    # and known optimal solutions, if any.
    input_files_dict = OrderedDict()
    for file in input_files:
        with open(file) as inp_f:
            csv_reader = csv.reader(inp_f, delimiter=',')
            graph_data = list()
            for i, row in enumerate(csv_reader):
                if i == 0:
                    # Number of vertices (cities) in the data
                    v = int(row[0])
                elif len(row) == 0:
                    # Omit the blank line
                    pass
                else:
                    # Extract rows from input file
                    graph_data.append(row)

        # Create a matrix containing the graph data, and assert it's shape
        graph_mat = np.array(graph_data[:v], dtype=np.float)
        if graph_mat.shape[0] != graph_mat.shape[1]:
            print("Error: The provided graph matrix is not square!")
            sys.exit(1)
        elif graph_mat.shape[0] != v:
            print("Error: No. of cities in the input file is not consistent with the ",
                  "graph matrix size")
            sys.exit(1)
        elif start_city is not None and start_city > v:
            print("Error: The start-city index should be in the range: [1, %d]." % v)
            sys.exit(1)

        input_files_dict[file] = OrderedDict()
        input_files_dict[file]['graph_mat'] = graph_mat
        input_files_dict[file]['v'] = v

        # Check if the input file contains any known optimal solution to compare against
        if len(graph_data) > v:
            optimal_solution = list()
            line = v
            while line < len(graph_data):
                if len(graph_data[line]) == 1:
                    # (start_city, optimal_path, cost)
                    optimal_solution.append((int(graph_data[line][0]),
                                             np.array(graph_data[line + 1], dtype=int).tolist(),
                                             float(graph_data[line + 2][0])))
                    line += 3
                else:
                    # (start_city = Any, optimal_path, cost)
                    optimal_solution.append((None, np.array(graph_data[line], dtype=int).tolist(),
                                             float(graph_data[line + 1][0])))
                    line += 2
            input_files_dict[file]['solutions'] = optimal_solution
        else:
            input_files_dict[file]['solutions'] = None

    return input_files_dict


def plot_graph_network(X, v, t=0):
    # Create an adjacency matrix of graph matrix X
    adj_mat = generate_adjacency_matrix(X)

    # Create a list of directed edges associated with vertices vᵢ and vⱼ, ∀ i, j ∈ {1, 2, ..., |V|}
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
    plt.waitforbuttonpress(t)
    plt.close()
    return

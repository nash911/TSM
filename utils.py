import numpy as np


def generate_adjacency_matrix(X):
    # Create an adjacency matrix of graph matrix X
    adj_mat = np.array(X >= 0, dtype=np.int)

    return adj_mat

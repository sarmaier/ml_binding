import os
import sys
import subprocess
import numpy as np
import pandas as pd
import json_numpy
import networkx as nx
from sklearn.model_selection import train_test_split


# Function to load data from a pickle
def load_json(name):
    with open(name, 'r') as file:
        return json_numpy.load(file)


# Function to calculate mean and standard deviation of node attributes
def get_mean_std(node_dictionary):
    labels = {0: 'vdw', 1: 'electrostatic', 2: 'polar_solv', 3: 'nonpolar_solv', 4: 'total', 5: 'ca_distance'}
    arrays = [node_dictionary[x] for x in node_dictionary]
    concatenated = np.concatenate(arrays)
    means_ = [np.mean(column) for column in concatenated.T.astype(float)][6:]
    means = {x: means_[x] for x in labels}
    stds_ = [np.std(column) for column in concatenated.T.astype(float)][6:]
    stds = {x: stds_[x] for x in labels}
    return means, stds


# Function to normalize node attributes
def normalize_nodes(array, means, stds):
    array = array.astype(float)
    half = array[:, 6:]
    for i, column in enumerate(half.T):
        norm = np.subtract(column, means[i])
        norm = np.divide(norm, stds[i]).T
        half[:, i] = norm
    normalized = np.concatenate([array[:, :6], half], axis=1)
    return normalized


# Function to create CSV files for edges and features
def make_csv(_pdb_id, graph):
    e = graph.edges
    features = np.array([graph.nodes[_node]['feature_array'] for _node in graph.nodes])
    n_features = len(graph.nodes[0]['feature_array'])
    edges_pd = pd.DataFrame(e, columns=['node_1', 'node_2'])
    features_pd = pd.DataFrame(features, columns=['x_' + str(i) for i in range(n_features)])
    edges_pd.to_csv(_pdb_id + "_edges.csv", index=False)
    features_pd.to_csv(_pdb_id + "_features.csv", index=False)


# Main functionality
if __name__ == "__main__":
    py_dir = os.path.dirname(os.path.abspath(sys.argv[0])) # src/ directory
    my_dir = os.getcwd()

    cmd = ["python", py_dir + "/build_gbsa_info.py"]
    subprocess.Popen(cmd).wait()
    adjacency = load_json("numpy_adjacency.json")
    nodes = load_json("numpy_nodes.json")

    # Splitting dataset into train and test
    train_pdb_ids, test_pdb_ids = train_test_split(list(adjacency), train_size=0.9)
    adjacency_train = {i: adjacency[i] for i in train_pdb_ids}
    nodes_train = {i: nodes[i] for i in train_pdb_ids}

    mean, std = get_mean_std(nodes_train)
    for pdb_id in nodes_train:
        print("Working on... " + str(pdb_id))
        normalized_nodes = normalize_nodes(nodes_train[pdb_id], mean, std)
        g = nx.from_numpy_array(adjacency_train[pdb_id])
        node_attributes = {idx: val for idx, val in enumerate(normalized_nodes, start=0)}
        for node in g.nodes:
            g.nodes[node]['feature_array'] = node_attributes[node]
        make_csv(pdb_id, g)
        feather_cmd = ["python", "../FEATHER/src/main.py", "--ls graph-input", pdb_id + "_edges.csv", "--feature-input",
                       pdb_id + "_features.csv", "--output", pdb_id + "_output.csv", "--model-type", "FEATHER-G-att"]
        subprocess.Popen(feather_cmd).wait()
        print("FEATHER output for . . . " + pdb_id + " finished!")

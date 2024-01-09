import os
import sys
import subprocess
import numpy as np
import pandas as pd
import json_numpy
import networkx as nx
from sklearn.model_selection import train_test_split


# Function to load data from a json
def load_json(name):
    with open(name, 'r') as file:
        return json_numpy.load(file)


def make_json(name, data):
    with open(name + '.json', 'w') as json_out:
        json_numpy.dump(data, json_out)


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
def make_feather_input(_pdb_id, graph):
    e = graph.edges
    features = np.array([graph.nodes[_node]['feature_array'] for _node in graph.nodes])
    n_features = len(graph.nodes[0]['feature_array'])
    edges_pd = pd.DataFrame(e, columns=['node_1', 'node_2'])
    features_pd = pd.DataFrame(features, columns=['x_' + str(i) for i in range(n_features)])
    edges_pd.to_csv(_pdb_id + "_edges.csv", index=False)
    features_pd.to_csv(_pdb_id + "_features.csv", index=False)


def make_feather(_id):
    feather_cmd = ["python", "../FEATHER/src/main.py", "--graph-input", _id + "_edges.csv", "--feature-input",
                   _id + "_features.csv", "--output", pdb_id + "_output.csv", "--model-type", "FEATHER-G-att"]
    subprocess.Popen(feather_cmd).wait()
    print("FEATHER output for . . . " + _id + " finished!")


# Main functionality
if __name__ == "__main__":
    # get src directory
    py_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    my_dir = os.getcwd()
    # execute script to build necessary data files
    cmd = ["python", py_dir + "/build_gbsa_info.py"]
    subprocess.Popen(cmd).wait()
    adjacency = load_json("numpy_adjacency.json")
    nodes = load_json("numpy_nodes.json")

    # Splitting dataset into train and test
    train_pdb_ids, test_pdb_ids = train_test_split(list(adjacency), train_size=0.9, random_state=42)
    make_json("pdb_ids_train", train_pdb_ids)
    # Get mean and standard deviation from training data
    nodes_train = {i: nodes[i] for i in train_pdb_ids}
    mean, std = get_mean_std(nodes_train)

    for pdb_id in nodes:
        print("Working on... " + str(pdb_id))
        normalized_nodes = normalize_nodes(nodes[pdb_id], mean, std)
        g = nx.from_numpy_array(adjacency[pdb_id])
        node_attributes = {idx: val for idx, val in enumerate(normalized_nodes, start=0)}
        for node in g.nodes:
            g.nodes[node]['feature_array'] = node_attributes[node]
        make_feather_input(pdb_id, g)
        make_feather(pdb_id)

    # combine features
    ligand_features = load_json("ligand_features.json")
    complex_features = load_json("complex_features.json")
    joined_features = {}
    no_pairwise_features = {}
    for pdb_id in ligand_features:
        complex_vec = complex_features[pdb_id]
        ligand_vec = ligand_features[pdb_id]
        feather_df = pd.read_csv(pdb_id + "_output.csv")
        feather_vec = feather_df.to_numpy()
        if ligand_vec[0] is not None and complex_vec[0] is not None:
            no_pairwise_vec = np.concatenate((complex_vec, ligand_vec), axis=1)
            no_pairwise_features[pdb_id] = no_pairwise_vec

            joined_vec = np.concatenate((complex_vec, ligand_vec, feather_vec), axis=1)
            joined_features[pdb_id] = joined_vec
        else:
            pass
    make_json("all_features", joined_features)
    make_json("no_pairwise_features", no_pairwise_features)

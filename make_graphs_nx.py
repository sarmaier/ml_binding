import glob
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from karateclub import FeatherGraph
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
np.random.seed(42)


def make_pickle(name, info):
    pickle_out = open(name + '.pickle', 'wb')
    pickle.dump(info, pickle_out)
    pickle_out.close()
    return


# read pickle file
def load_pickle(name):
    x = pickle.load(open(name, 'rb'))
    return x


def get_mean_std(node_dictionary):
    labels = {0: 'vdw', 1: 'electrostatic', 2: 'polar_solv', 3: 'nonpolar_solv', 4: 'total', 5: 'ca_distance'}
    arrays = [node_dictionary[x] for x in node_dictionary]
    concatenated = np.concatenate(arrays)
    means_ = [np.mean(column) for column in concatenated.T.astype(float)][6:]
    means = {x: means_[x] for x in labels}
    stds_ = [np.std(column) for column in concatenated.T.astype(float)][6:]
    stds = {x: stds_[x] for x in labels}
    return means, stds


def normalize_nodes(array, m, s):
    array = array.astype(float)
    half = array[:, 6:]
    for i, column in enumerate(half.T):
        norm = np.subtract(column, m[i])
        norm = np.divide(norm, s[i]).T
        half[:, i] = norm
    normalized = np.concatenate([array[:, :6], half], axis=1)
    return normalized


def make_csv(id_, graph):
    e = graph.edges
    features = np.array([graph.nodes[node]['feature_array'] for node in graph.nodes])
    n_features = len(graph.nodes[0]['feature_array'])
    edges_pd = pd.DataFrame(e, columns=['node_1', 'node_2'])
    features_pd = pd.DataFrame(features, columns=['x_' + str(i) for i in range(n_features)])
    edges_pd.to_csv(id_ + "_edges_11_30_23.csv", index=False)
    features_pd.to_csv(id_ + "_features_11_30_23.csv", index=False)
    return


# Example usage
if __name__ == "__main__":
    # Example: List of NetworkX graphs with attributes
    adjacency = load_pickle("numpy_adjacency_11_15_23.pickle")
    nodes = load_pickle("numpy_nodes_11_15_23.pickle")
    # split dataset into train and test
    train_pdb_ids, test_pdb_ids = train_test_split(list(adjacency), train_size=0.9)
    adjacency_train = {i: adjacency[i] for i in train_pdb_ids}
    nodes_train = {i: nodes[i] for i in train_pdb_ids}
    mean, std = get_mean_std(nodes_train)
    for pdb_id in nodes_train:
#    for pdb_id in ['2qmg']:
        print("working on . . . " + str(pdb_id))
        normalized_nodes = normalize_nodes(nodes_train[pdb_id], mean, std)
        # make networkx graph from numpy adjacency matrix
        g = nx.from_numpy_array(adjacency_train[pdb_id])
        # update dictionary of node attributes with labels
        node_att = {idx: val for idx, val in enumerate(normalized_nodes, start=0)}
        for node in g.nodes:
            g.nodes[node]['feature_array'] = node_att[node]
        make_csv(pdb_id, g)






    #nx.draw(g)
    #plt.show()

#make_pickle('neural_net_features_09_20_23', feature_dict)
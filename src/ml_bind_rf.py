import os
import sys
from datetime import date
import glob
import json_numpy
import logging



# Function to load data from a json
def load_json(name):
    with open(name, 'r') as file:
        return json_numpy.load(file)



if __name__ == "__main__":
    # get src directory
    py_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    my_dir = os.getcwd()
    # execute script to build necessary data files
    ligand_features = load_json("ligand_features.json")
    print(ligand_features)
    exit()

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
        feather_cmd = ["python", "../FEATHER/src/main.py", "--graph-input", pdb_id + "_edges.csv", "--feature-input",
                       pdb_id + "_features.csv", "--output", pdb_id + "_output.csv", "--model-type", "FEATHER-G-att"]
        subprocess.Popen(feather_cmd).wait()
        print("FEATHER output for . . . " + pdb_id + " finished!")
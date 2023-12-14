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
    complex_features = load_json("complex_features.json")
    pdb_ids = [x for x in ligand_features]

    exit()


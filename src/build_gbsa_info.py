import os
from datetime import date
import glob
import json_numpy
from ligand_block import LigandBlock
from gbsa_block import GbsaComplexBlock
from interaction_gbsa_block import GbsaInteraction
import logging

logging.basicConfig(filename='processing_logs.log', level=logging.ERROR)


def make_json(name, data):
    with open(name + '.json', 'w') as json_out:
        json_numpy.dump(data, json_out)


class GenInfoError(Exception):
    """Custom exception for error handing generate_rooted_info.py"""
    def __init__(self, message="error processing file. "):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    my_dir = os.getcwd()  # current directory
    pdb_paths = [x for x in glob.glob(my_dir + "/*_ligand.xyz")]
    pdb_ids = [os.path.split(path_name)[1].split("_ligand.xyz")[0] for path_name in pdb_paths]
    ligand_dict, complex_dict = {}, {}
    edges_dict, adjacency_dict, nodes_dict = {}, {}, {}

    for pdb_id in pdb_ids:
        print("Working on... " + str(pdb_id))
        try:
            ligand_block = LigandBlock(pdb_id)
            ligand_features = ligand_block.ligand_features
            ligand_dict[pdb_id] = ligand_features
        except Exception:
            ligand_dict[pdb_id] = ['None']
        try:
            gbsa_block = GbsaComplexBlock(pdb_id)
            complex_features = gbsa_block.bind_properties
            complex_dict[pdb_id] = complex_features
        except Exception:
            complex_dict[pdb_id] = ['None']
        try:
            interaction_block = GbsaInteraction(pdb_id)
            edges_dict[pdb_id] = interaction_block.edges
            adjacency_dict[pdb_id] = interaction_block.adjacency
            nodes_dict[pdb_id] = interaction_block.nodes
        except GenInfoError as e:
            print("Error caught with interaction_block for PDB ID-->" + pdb_id + ": ", e.message)
            exit()

    today = date.today()
    make_json('ligand_features', ligand_dict)
    make_json('complex_features', complex_dict)
    make_json('numpy_adjacency', adjacency_dict)
    make_json('numpy_edges', edges_dict)
    make_json('numpy_nodes', nodes_dict)



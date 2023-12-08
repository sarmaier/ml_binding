import os
from datetime import date
import glob
import json_numpy
from interaction_gbsa_block import GbsaInteraction
import logging

logging.basicConfig(filename='processing_logs.log', level=logging.ERROR)


def make_json(name, data):
    with open(name + '.json', 'w') as json_out:
        json_numpy.dump(data, json_out)


if __name__ == "__main__":
    my_dir = os.getcwd()  # current directory
    pdb_paths = [x for x in glob.glob(my_dir + "/3n*_ligand.xyz")]
    pdb_ids = [os.path.split(path_name)[1].split("_ligand.xyz")[0] for path_name in pdb_paths]
    edges_dict, adjacency_dict, nodes_dict = {}, {}, {}

    for pdb_id in pdb_ids:
        print("Working on... " + str(pdb_id))
        try:
            interaction_block = GbsaInteraction(pdb_id)
            edges_dict[pdb_id] = interaction_block.edges
            adjacency_dict[pdb_id] = interaction_block.adjacency
            nodes_dict[pdb_id] = interaction_block.nodes

        except FileNotFoundError:
            logging.error(f"File not found for {pdb_id}")
        except Exception as e:
            logging.error(f"Error processing {pdb_id}: {str(e)}")
            exit()
    today = date.today()
    make_json('numpy_adjacency', adjacency_dict)
    make_json('numpy_edges', edges_dict)
    make_json('numpy_nodes', nodes_dict)

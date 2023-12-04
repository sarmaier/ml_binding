import traceback
from datetime import date
import glob
import numpy as np
import pickle
from interaction_gbsa_block import GbsaInteraction
import logging

logging.basicConfig(filename='processing_logs.log', level=logging.ERROR)


def make_pickle(name, info):
    pickle_out = open(name + '.pickle', 'wb')
    pickle.dump(info, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    pdb_ids = [x.split("_ligand.xyz")[0] for x in glob.glob("*ligand.xyz")]
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
    make_pickle('numpy_adjacency_' + str(today), adjacency_dict)
    make_pickle('numpy_edges_' + str(today), edges_dict)
    make_pickle('numpy_nodes_' + str(today), nodes_dict)
import glob
import numpy as np
import pickle
from make_interaction_gbsa_block import GbsaInteraction
import networkx as nx
import matplotlib.pyplot as plt


def make_pickle(name, info):
    pickle_out = open(name + '.pickle', 'wb')
    pickle.dump(info, pickle_out)
    pickle_out.close()
    return


# Example usage
if __name__ == "__main__":
    # Example: List of NetworkX graphs with attributes
    pdb_ids = [x.split("_stripped_complex_ambpdb_08_28_23_ligand.xyz")[0] for x in glob.glob("*ligand.xyz")]
    edges_dict, adjacency_dict, nodes_dict = {}, {}, {}
    for pdb_id in pdb_ids:
        print("working on . . . " + str(pdb_id))
        # call classes to get numpy blocks for each pdb id
        try:
            interaction_block = GbsaInteraction(pdb_id)
            edges_dict[pdb_id] = interaction_block.edges
            adjacency_dict[pdb_id] = interaction_block.adjacency
            nodes_dict[pdb_id] = interaction_block.nodes

        except Exception:
            print("Error processing " + str(pdb_id) + " numpy matrices!")
            exit()

    make_pickle('numpy_adjacency_11_15_23', adjacency_dict)
    make_pickle('numpy_edges_11_15_23', edges_dict)
    make_pickle('numpy_nodes_11_15_23', nodes_dict)
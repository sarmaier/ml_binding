from datetime import date
import glob
import json
from interaction_gbsa_block import GbsaInteraction
import logging

logging.basicConfig(filename='processing_logs.log', level=logging.ERROR)


def make_json(name, info):
    with open(name + '.json', 'w') as json_out:
        json.dump(info, json_out)


if __name__ == "__main__":
    pdb_ids = [x.split("_ligand.xyz")[0] for x in glob.glob("3nkk_ligand.xyz")]
    edges_dict, adjacency_dict, nodes_dict = {}, {}, {}

    for pdb_id in pdb_ids:
        print("Working on... " + str(pdb_id))
        try:
            interaction_block = GbsaInteraction(pdb_id)
            edges_dict[pdb_id] = interaction_block.edges.tolist()
            adjacency_dict[pdb_id] = interaction_block.adjacency.tolist()
            nodes_dict[pdb_id] = interaction_block.nodes.tolist()

        except FileNotFoundError:
            logging.error(f"File not found for {pdb_id}")
        except Exception as e:
            logging.error(f"Error processing {pdb_id}: {str(e)}")
            exit()
    today = date.today()
    make_json('numpy_adjacency_' + str(today), adjacency_dict)
    make_json('numpy_edges_' + str(today), edges_dict)
    make_json('numpy_nodes_' + str(today), nodes_dict)

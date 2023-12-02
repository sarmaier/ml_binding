import glob
import numpy as np
import pickle
from make_ligand_block import LigandBlock
from make_interaction_gbsa_block import GbsaInteraction
from make_complex_gbsa_block import GbsaComplexBlock


def make_pickle(name, info):
    pickle_out = open(name + '.pickle', 'wb')
    pickle.dump(info, pickle_out)
    pickle_out.close()
    return


files = [x.split("_stripped_complex_ambpdb_08_28_23_ligand.xyz")[0] for x in glob.glob("*ligand.xyz")]
feature_dict = {}
for pdb in files:
#for pdb in ['3t01']:
    print(pdb)
    # call classes to make features for each block
    ligand_block = LigandBlock(pdb)
    complex_block = GbsaComplexBlock(pdb)
    interaction_block = GbsaInteraction(pdb)

    persistence_features = interaction_block.persistence_features
    bind_features = complex_block.bind_properties
    complex_features = np.concatenate((persistence_features, bind_features), axis=0)
    ligand_features = ligand_block.ligand_features
    total_features = np.concatenate((complex_features, ligand_features), axis=0)
    total_features = np.reshape(total_features, (1, 39))
    feature_dict[pdb] = total_features

    # nodes = interaction_block.nodes
    # adjacency = interaction_block.adjacency
    # edges = interaction_block.edges
make_pickle('neural_net_features_09_20_23', feature_dict)

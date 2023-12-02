import glob
import numpy as np
import pickle
from make_ligand_block import LigandBlock
from make_interaction_gbsa_block import GbsaInteraction
from make_complex_gbsa_block import GbsaComplexBlock


def make_pickle(name, info):
    with open(name + '.pickle', 'wb') as pickle_out:
        pickle.dump(info, pickle_out)


files = [x.split("_stripped_complex_ambpdb_08_28_23_ligand.xyz")[0] for x in glob.glob("*ligand.xyz")]
feature_dict = {}

for pdb in files:
    try:
        print(f"Processing {pdb}")
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

    except Exception as e:
        print(f"Error processing {pdb}: {e}")

make_pickle('neural_net_features_09_20_23', feature_dict)
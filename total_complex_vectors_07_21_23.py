import os
import sys
import re
import math
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from gtda.diagrams import BettiCurve, PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
import category_encoders as ce
from scipy.spatial import distance
from operator import itemgetter


def make_pickle(name, info):
    pickle_out = open(str(name) + '.pickle', 'wb')
    pickle.dump(info, pickle_out)
    pickle_out.close()
    return


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_string(filename):
    string = open(filename, "r").read()
    return string


def get_exp(string, pdb_names):
    lines = [line.strip('\n') for line in string.split('\n') if line][6:]
    name = [[x for x in line.split(' ') if x][0] for line in lines if line]
    pk = [[x for x in line.split(' ') if x][3] for line in lines if line]
    pk = {name[i]: pk[i] for i in range(len(name))}
    pk_finished = [float(pk[i]) for i in pdb_names]
    R = 0.001987
    T = 298.15
    exp_values = [R * T * math.log(10 ** (-x)) for x in pk_finished]
    exp_values = [round(x, 4) for x in exp_values]
    exp_values = [np.reshape(np.array(x), (1,)) for x in exp_values]
    exp_values = np.array(exp_values)
    return exp_values


def gbsa_complex(string):
    com = [x for x in re.findall(r"Complex:(.*?)Receptor:", string, re.DOTALL)[0].split("\n") if x][2:]
    com = [[float(x) for x in y.split(" ") if "." in x][0] for y in com]
    return com


def gbsa_receptor(string):
    rec = [x for x in re.findall(r"Receptor:(.*?)Ligand:", string, re.DOTALL)[0].split("\n") if x][2:]
    rec = [[float(x) for x in y.split(" ") if "." in x][0] for y in rec]
    return rec


def gbsa_ligand(string):
    lig = [x for x in re.findall(r"Ligand:(.*?)Difference", string, re.DOTALL)[0].split("\n") if x][2:]
    lig = [[float(x) for x in y.split(" ") if "." in x][0] for y in lig]
    return lig


def gbsa_delta(string):
    delta = [x for x in re.findall(r"Difference(.*?)\n\n------", string, re.DOTALL)[0].split("\n") if x][3:]
    delta = [[float(x) for x in y.split(" ") if "." in x][0] for y in delta]
    return delta


def add_residue_mass(pairs_list):
    masses = {"gly": 57.051, "ala": 71.078, "ser": 87.077, "pro": 97.115, "val": 99.131, "thr": 101.104,
              "cys": 103.143, "ile": 113.158, "leu": 113.158, "asn": 114.103, "asp": 115.087, "gln": 128.129,
              "lys": 128.172, "glu": 129.114, "met": 131.196, "hie": 137.139, "phe": 147.174, "arg": 156.186,
              "tyr": 163.173, "trp": 186.21, "cyx": 206.286}
    for i in range(len(pairs_list)):
        x = pairs_list[i][:]
        x.insert(1, masses[pairs_list[i][0]])
        pairs_list[i] = x
    return


def topo_feature(xyz_f):
    xyz = xyz_f.split("\n")[2:]
    xyz = [float(x) for y in xyz for x in y.split(" ")[1:] if x]
    n_atoms = len(xyz)
    xyz = [np.array(xyz[i:i + 3]) for i in range(0, n_atoms, 3)]
    xyz = np.array(xyz)
    dist_mat = pdist(xyz)
    dist_mat = squareform(dist_mat)
    dist_mat = dist_mat.reshape(1, *dist_mat.shape)
    VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1])
    diagrams = VR.fit_transform(dist_mat)
    PE = PersistenceEntropy()
    topo_pe = PE.fit_transform(diagrams)
    topo = np.reshape(topo_pe.flatten(), (1, 2))
    return topo


def encode_residues(pairs_list):
    residues = ["gly", "ala", "ser", "pro", "val", "thr", "cys", "ile", "leu", "asn", "asp",
                "gln", "lys", "glu", "met", "hie", "phe", "arg", "tyr", "trp", "cyx"]
    res_name = pd.DataFrame([x[0] for x in pairs_list], columns=['residue_name'])
    residues = pd.DataFrame(residues, columns=['residue_name'])
    encoder = ce.ordinal.OrdinalEncoder(cols='residue_name')
    encoder.fit(residues)
    res_rep = encoder.transform(res_name)['residue_name'].values.tolist()
    for i in range(len(pairs_list)):
        pairs_list[i][0] = res_rep[i]
    encoded_pairs = pairs_list
    return encoded_pairs


pdb_names_pairwise = [x for x in get_string("gbsa_pairwise_finished_names.txt").split("\n") if x]
pdb_names_total = [x for x in get_string("gbsa_finished_names.txt").split("\n") if x]
pdb_names_active = [x for x in get_string("active_finished_names.txt").split("\n") if x]
pdb_names_xtb_l = [x.split(":")[0] for x in get_string("ligand.energies").split("\n")]
pdb_names_xtb_p = [x.split(":")[0] for x in get_string("protein.energies").split("\n")]
pdb_names_xtb_c = [x.split(":")[0] for x in get_string("complex.energies").split("\n")]
name_lists = [pdb_names_total, pdb_names_active, pdb_names_xtb_l, pdb_names_xtb_p, pdb_names_xtb_c]
name_lists = set.intersection(*map(set, name_lists))
pdb_names = [x for x in pdb_names_pairwise if x in name_lists]
exp_txt = get_string("INDEX_refined_data.2020")
exp_values = get_exp(exp_txt, pdb_names)
features = []
for name in pdb_names:
    print(name)
    _ = []

    # get xtb binding energy
    xtb_diff = xtb_diff_terms(name)
    # xtb_bind = xtb_energy(name)

    # process residue-ligand pairwise interactions
    pair_str = get_string(name + "_ligand_interaction.txt")
    parsed_pairs = parse_pairwise(pair_str)
    add_residue_mass(parsed_pairs)
    joined_pairs = encode_interaction(parsed_pairs)
    #	parsed_pairs = encode_residues(parsed_pairs)
    #	res_10 = top_residues_dist(name, parsed_pairs)
    #	parsed_pairs = [parsed_pairs[x] for x in res_10]
    #	res_10 = top_residues_e(name, parsed_pairs)
    #	parsed_pairs = res_10
    #	joined_pairs = np.array(parsed_pairs).flatten()
    #	if len(joined_pairs) < 140:
    #		joined_pairs = np.pad(joined_pairs, (0, 70 - len(joined_pairs)))
    #	joined_pairs = np.reshape(joined_pairs, (1,70))

    # process information from total interaction energy
    gbsa_str = get_string("FINAL_RESULTS_MMGBSA_" + name + ".dat")
    com = gbsa_complex(gbsa_str)
    rec = gbsa_receptor(gbsa_str)
    lig = gbsa_ligand(gbsa_str)
    delta = gbsa_delta(gbsa_str)
    _.extend(com)
    _.extend(rec)
    _.extend(lig)
    _.extend(delta)
    tot_int = np.reshape(np.array(_), (1, 28))

    # process structural information using persistant topology descriptor
    ligand_xyz_f = get_string(name + "_ligand_amber.xyz")
    ligand_topo = topo_feature(ligand_xyz_f)
    protein_pdb_f = get_string(name + "_min3_pdb_pdb4amber_active.pdb")
    pdb_2_xyz(name, protein_pdb_f)
    complex_xyz_f = get_string(name + "_complex_amber.xyz")
    complex_topo = topo_feature(complex_xyz_f)

    # join all feature types
    joined_feature = np.concatenate((xtb_diff, tot_int, ligand_topo, complex_topo, joined_pairs), axis=1)
    # joined_feature = np.concatenate((tot_int, ligand_topo, complex_topo), axis=1)
    features.append(joined_feature)

features = np.array(features)
features = np.reshape(features, (len(features), 171))
# features = np.reshape(features, (len(features),32))
make_pickle("gbsa_features_w_xtb_terms", features)
make_pickle("exp_y_w_xtb_terms", exp_values)

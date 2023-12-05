import os
import re
import numpy as np
from datetime import date
import time
import argparse
from scipy.spatial.distance import pdist, squareform
from gtda.diagrams import BettiCurve, PersistenceEntropy
from gtda.homology import VietorisRipsPersistence


def compute_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def alpha_c_distance(pdb_list):
    ca_list = [[x for x in y.split(" ") if x] for y in pdb_list if " CA " in y and "AT" in y]
    mol_list = [[x for x in y.split(" ") if x] for y in pdb_list if "MOL" in y and "AT" in y]
    ca_xyz = {x[3].lower() + "_" + x[4]: np.array(x[5:8]).astype('float64') for x in ca_list}
    mol_xyz = {x[3].lower() + "_" + x[1]: np.array(x[5:8]).astype('float64') for x in mol_list}
    ca_lig_distances = {}
    for atom in ca_xyz:
        all_dist = [compute_distance(ca_xyz[atom], mol_xyz[x]) for x in mol_xyz]
        if any(x <= 8 for x in all_dist):
            ca_lig_distances[atom] = round(min(all_dist), 4)
    return ca_xyz, ca_lig_distances


def parse_pairwise(str_):
    aa = dict.fromkeys(["arg", "hip", "lys"], [0, 1, 0, 0, 0, 0])
    aa.update(dict.fromkeys(["asp", "glu"], [0, 0, 1, 0, 0, 0]))
    aa.update(dict.fromkeys(["ser", "thr", "asn", "gln"], [0, 0, 0, 1, 0, 0]))
    aa.update(dict.fromkeys(["cys", "sec", "gly", "pro", "cyx"], [0, 0, 0, 0, 1, 0]))
    aa.update(dict.fromkeys(["ala", "ile", "leu", "met", "phe", "trp", "tyr", "val", "hie", "hid"], [0, 0, 0, 0, 0, 1]))
    str_ = "   1,".join(str_.split("   1,")[0:2])
    _ = [[x for x in y.split(",")] for y in str_.split("\n")[:-1] if y]
    for x in _:
        match1 = re.match(r"([a-zA-Z]+)([0-9]+)", x[0])
        if match1:
            x[0] = "_".join(match1.groups())
        else:
            x[0] = re.sub(r"\s+", "_", x[0])
        match2 = re.match(r"([a-zA-Z]+)([0-9]+)", x[1])
        if match2:
            x[1] = "_".join(match2.groups())
        else:
            x[1] = re.sub(r"\s+", "_", x[1])
    _ = [y[0:2] + y[5::3] for y in _ if y]
    parsed_str = [[round(float(x), 4) if '.' in x else x.lower() for x in y] for y in _]
    parsed_int = {x[0]: aa[x[0].split("_")[0]] + x[2:] for x in parsed_str}
    return parsed_int


class ProcessingError(Exception):
    pass


class GbsaInteraction:
    def __init__(self, _id):
        int_string = open("FINAL_RESULTS_MMGBSA_per_residue_" + _id + "_interaction.txt").read()
        pdb_list = open(_id + "_complex.pdb", "r").read().split('\n')
        self.int = parse_pairwise(int_string)
        self.ca_coord, self.alpha_dist = alpha_c_distance(pdb_list)
        self.n_ca = len(list(self.alpha_dist))
        self.ca_index = {x: list(self.alpha_dist)[x] for x in range(self.n_ca)}
        self.edges, self.distance_matrix = self.make_edges()
        self.adjacency = self.make_adjacency()
        self.nodes = self.make_nodes()
        self.persistence_features = self.get_persistence_features()

    def make_edges(self):
        n = self.n_ca
        index = self.ca_index
        ca_xyz = self.ca_coord
        ca_dist = self.alpha_dist
        xyz_array = np.array([ca_xyz[index[x]] for x in range(n)])
        e = squareform(pdist(xyz_array))
        full_dist_matrix = e.copy()
        e[e > 5.0] = 0.0
        indexed_mol_d = np.array([ca_dist[index[x]] for x in index] + ['0.0'])
        e = np.vstack([e, np.zeros((1, n))])
        e = np.hstack([e, np.zeros((n + 1, 1))])
        e[-1, :] = indexed_mol_d
        e[:, -1] = indexed_mol_d.T
        return e, full_dist_matrix

    def make_adjacency(self):
        ad = np.copy(self.edges)
        n_col = ad.shape[1]
        ad[np.where((ad <= 5.0) & (ad > 0))] = 1.0
        ad[ad > 5.0] = 0.0
        ad[-1, :] = np.ones((1, n_col))
        ad[:, -1] = np.ones((n_col,))
        #np.fill_diagonal(ad, 1)
        return ad

    def make_nodes(self):
        index = self.ca_index
        nodes = {x: self.int[x] + [self.alpha_dist[x]] for x in self.alpha_dist}
        nodes = np.array([nodes[index[x]] for x in index])
        nodes = np.vstack((nodes, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
        return nodes

    def get_persistence_features(self):
        matrix = self.distance_matrix
        ca_matrix = matrix.reshape(1, *matrix.shape)
        vr = VietorisRipsPersistence(metric='precomputed', homology_dimensions=[0, 1])
        diagrams = vr.fit_transform(ca_matrix)
        features = PersistenceEntropy().fit_transform(diagrams).flatten()
        return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("PDB_ID", help="PDB ID of the molecule")
    args = parser.parse_args()

    pdb_id = args.PDB_ID
    if not os.path.isfile(f"FINAL_RESULTS_MMGBSA_per_residue_{pdb_id}_interaction.txt"):
        print(f"Required files not found for PDB ID {pdb_id}")
    else:
        try:
            complex_gbsa = GbsaInteraction(pdb_id)
        except ProcessingError:
            print(f"Issue with processing {pdb_id}")
            exit()

    today = date.today()
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print("FINISHED PROCESSING\n" + str(current_time))
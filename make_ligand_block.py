import os
import sys
import re
import math
import numpy as np
import pandas as pd
from datetime import date
import time
import argparse
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial import distance
np.set_printoptions(threshold=sys.maxsize)
from xyz2mol import xyz2mol
from xyz2mol import read_xyz_file
from rdkit.Chem import rdmolfiles
from rdkit.Chem import Draw
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors


def get_mol(filename):
    a_num, charge, xyz_coord = read_xyz_file(filename)
    charged_fragments = True
    quick = True
    huckel = False
    mol_object = xyz2mol(a_num, charge, xyz_coord, charged_fragments, quick, huckel)
    #images = Draw.MolsToImage([mol_object], subImgSize=(700,700))
    #images.save('test.png')
    return mol_object


def get_xtb_features(filename):
    xtb_str = open(filename, "r").read()
    homo_e = [x for x in xtb_str.split("(HOMO)")[-2].split(" ") if x][-1]
    lumo_e = [x for x in xtb_str.split("(LUMO)")[-2].split(" ") if x][-1]
    total_e = [x for x in xtb_str.split("TOTAL ENERGY")[-1].split("Eh")[0].split(" ") if x][-1]
    grad_norm = [x for x in xtb_str.split("GRADIENT NORM")[-1].split("Eh")[0].split(" ") if x][-1]
    homo_lumo_gap = [x for x in xtb_str.split("HOMO-LUMO GAP")[-1].split("eV")[0].split(" ") if x][-1]
    iso_es = [x for x in xtb_str.split("isotropic ES")[-1].split("Eh")[0].split(" ") if x][-1]
    aniso_es = [x for x in xtb_str.split("anisotropic ES")[-1].split("Eh")[0].split(" ") if x][-1]
    aniso_xc = [x for x in xtb_str.split("anisotropic XC")[-1].split("Eh")[0].split(" ") if x][-1]
    dispersion = [x for x in xtb_str.split("dispersion")[-1].split("Eh")[0].split(" ") if x][-1]
    repulsion = [x for x in xtb_str.split("repulsion energy")[-1].split("Eh")[0].split(" ") if x][-1]
    dipole = [x for x in xtb_str.split("\nmolecular quadrupole")[0].split(" ") if x][-1]
    return np.array([homo_e, lumo_e, total_e, grad_norm, homo_lumo_gap, iso_es, aniso_es, aniso_xc,
                     dispersion, repulsion, dipole]).astype(float)

class LigandBlock:

    def __init__(self, id):
        xyz_file = id + "_stripped_complex_ambpdb_08_28_23_ligand.xyz"
        xtb_file = id + "_stripped_complex_ambpdb_08_28_23_ligand.out"
        mol = get_mol(xyz_file)

        n_h_acceptors = Lipinski.NumHAcceptors(mol)
        n_h_donors = Lipinski.NumHDonors(mol)
        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        n_aliphatic_rings = Lipinski.NumAliphaticRings(mol)
        n_aromatic_rings = Lipinski.NumAromaticRings(mol)
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        rdkit_features = np.array([n_h_acceptors, n_h_donors, rotatable_bonds, n_aliphatic_rings,
                                   n_aromatic_rings, mol_weight]).astype(float)
        xtb_features = get_xtb_features(xtb_file)
        self.ligand_features = np.concatenate((rdkit_features, xtb_features), axis=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("PDB ID")
    args = parser.parse_args()
    species = [getattr(args, arg) for arg in vars(args)]
    present_species = []
    mmgbsa_total_complex = {}
    for i in species:
        try:
            assert os.path.isfile("FINAL_RESULTS_MMGBSA_" + i + ".dat")
            assert os.path.isfile("FINAL_RESULTS_MMGBSA_per_residue_" + i + "_interaction.txt")
            assert os.path.isfile(i + "_stripped_complex_ambpdb.pdb")
            assert os.path.isfile(i + "_stripped_complex_ambpdb_08_28_23_ligand.xyz")
            assert os.path.isfile(i + "_stripped_complex_ambpdb_08_28_23_ligand.out")
            present_species.append(i)
        except AssertionError:
            pass
    for pdb_id in present_species:
        try:
            complex_gbsa = LigandBlock(pdb_id)
        except Exception:
            print("Issue with FINAL_RESULTS_MMGBSA_per_residue_" + id + ".dat")
            exit()
    today = date.today()
    time = time.strftime("%H:%M:%S", time.localtime())
    print(i + ":FINISHED PROCESSING")
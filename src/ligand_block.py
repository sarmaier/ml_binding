import os
import sys
import re
import numpy as np
from datetime import date
import time
import argparse
from rdkit.Chem import Lipinski, rdMolDescriptors
from xyz2mol import xyz2mol, read_xyz_file

np.set_printoptions(threshold=sys.maxsize)


def get_molecule_properties(filename):
    a_num, charge, xyz_coord = read_xyz_file(filename)
    charged_fragments = True
    quick = True
    huckel = False
    mol_object = xyz2mol(a_num, charge, xyz_coord, charged_fragments, quick, huckel)
    return mol_object


def extract_single_property(regex_pattern, xtb_str, default=0.0):
    match = re.search(regex_pattern, xtb_str)
    print(match)
    return float(match.group(1)) if match else default


def extract_xtb_features(xtb_str):
    properties = [
        extract_single_property(r'(-?\d+?\.\d+)\s+\(HOMO\)', xtb_str),
        extract_single_property(r'(-?\d+?\.\d+)\s+\(LUMO\)', xtb_str),
        extract_single_property(r'TOTAL ENERGY\s+(-?\d+?\.\d+)\s+?Eh', xtb_str),
        extract_single_property(r'GRADIENT NORM\s+(-?\d+?\.\d+)\s+?Eh', xtb_str),
        extract_single_property(r'HOMO-LUMO GAP\s+(-?\d+?\.\d+)\s+?eV', xtb_str),
        extract_single_property(r'-> isotropic ES\s+(-?\d+?\.\d+)\s+?Eh', xtb_str),
        extract_single_property(r'-> anisotropic ES\s+(-?\d+?\.\d+)\s+?Eh', xtb_str),
        extract_single_property(r'-> anisotropic XC\s+(-?\d+?\.\d+)\s+?Eh', xtb_str),
        extract_single_property(r'-> dispersion\s+(-?\d+\.\d+?)\s+?Eh', xtb_str),
        extract_single_property(r'repulsion energy\s+(-?\d+?\.\d+)\s+?Eh', xtb_str),
        extract_single_property(r'(-?\d+?\.\d+)\nmolecular quadrupole \(traceless\)', xtb_str)
    ]
    return np.array(properties)


class ProcessingError(Exception):
    pass


class LigandBlock:
    def __init__(self, pdb_id):
        xyz_file = f"{pdb_id}_ligand.xyz"
        xtb_file = f"{pdb_id}_ligand_xtb2.out"

        mol = get_molecule_properties(xyz_file)
        rdkit_features = np.array([
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumAliphaticRings(mol),
            Lipinski.NumAromaticRings(mol),
            rdMolDescriptors.CalcExactMolWt(mol)
        ]).astype(float)

        xtb_features = extract_xtb_features(open(xtb_file, "r").read())
        self.ligand_features = np.concatenate((rdkit_features, xtb_features), axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("PDB_ID")
    args = parser.parse_args()
    pdb_ids = [getattr(args, arg) for arg in vars(args)]
    print(pdb_ids)
    present_pdb_ids = []
    for pdb_id in pdb_ids:
        try:
            assert os.path.isfile(f"FINAL_RESULTS_MMGBSA_{pdb_id}.dat")
            assert os.path.isfile(f"FINAL_RESULTS_MMGBSA_per_residue_{pdb_id}_interaction.txt")
            assert os.path.isfile(f"{pdb_id}_complex.pdb")
            assert os.path.isfile(f"{pdb_id}_ligand.xyz")
            assert os.path.isfile(f"{pdb_id}_ligand_xtb2.out")
            present_pdb_ids.append(pdb_id)
        except AssertionError:
            print("Required files not found in path")
            pass

    for pdb_id in present_pdb_ids:
        try:
            complex_gbsa = LigandBlock(pdb_id)
        except ProcessingError:
            print(f"Issue with processing {pdb_id}")
            exit()

    today = date.today()
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Processed PDB IDs: {', '.join(present_pdb_ids)}")
    print("FINISHED PROCESSING")

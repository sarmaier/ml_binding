import sys
import subprocess
import numpy as np
from rdkit import Chem
from xyz2mol import xyz2mol
from rdkit import Chem
from rdkit.Chem import Draw


# --------------------------------------------------------------------------------#
# -----> Converts log/xyz file to RDKIT mol object using OpenBabel          <-----#
# --------------------------------------------------------------------------------#

class MolCbh:
    """ class to give you mol object as made by RDKIT. This is to be used
    when constructing CBH fragments
    """

    @staticmethod
    def get_labels(string):
        """get atom type as labelled in Gaussian log file. Returns dictionaries"""
        lst = list(filter(None, string.split(
            'Number     Number       Type             X           Y           Z\n '
            '---------------------------------------------------------------------\n      '
            )[1].split('\n ---')[0].split('\n')))
        lst = [int(list(filter(None, x.split(' ')))[1]) for x in lst]
        labels = {i: lst[i] for i in range(len(lst)) if lst[i] != 1}
        return labels

    @staticmethod
    def byte_2_string(byte):
        """convert byte output to string"""
        return byte.decode('UTF-8').rstrip()

    def log_2_xyz(self):
        """get xyz coordinates from Gaussian log file"""
        atomic_num = {'H': 1,'B':5,'C': 6, 'N': 7, 'O': 8,'F':9,'Si':14,'P':15,'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
        byte_output = subprocess.check_output('obabel ' + str(self.filename) + ' -oxyz', shell=True)
        # list of lists which each element containing atom as 0th element and x,y,z as 1st:3rd element
        _ = [[x for x in list(filter(None, y.split(' ')))] for y in self.byte_2_string(byte_output).split('\n')[2:]]
        # dictionary of atom identity
        labels = {x: atomic_num[_[x][0]] for x in range(len(_))}
        # dictionary of atom coordinates
        coordinates = {x: [x for x in _[x][1:]] for x in range(len(_))}
        return byte_output, labels, coordinates

    @staticmethod
    def get_xyz(f_input):
        """get xyz coordinates from .xyz file. Returns dictionaries"""
        atomic_num = {'H': 1,'B':5,'C': 6, 'N': 7, 'O': 8,'F':9,'Si':14,'P':15,'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
        lines = [line.strip() for line in open(f_input, 'r')][2:]
        _ = [[x for x in list(filter(None, y.split(' ')))] for y in lines]
        labels = {x: atomic_num[_[x][0]] for x in range(len(_))}
        coordinates = {x: [x for x in _[x][1:]] for x in range(len(_))}
        return labels, coordinates

    def get_mol(self, labels, coordinates):
        """get RDKIT mol object using xyz2mol.py from Jensen group. Returns RDKIT mol object"""
        coordinates = [[float(x) for x in y] for y in coordinates.values()]
        mol = xyz2mol(list(labels.values()), coordinates, use_graph=True, allow_charged_fragments=True,
                      embed_chiral=False, use_huckel=True)
        
#        Chem.SanitizeMol(mol)
#        Chem.Kekulize(mol)
#        formal_charges = {atom.GetIdx(): atom.GetFormalCharge() for atom in mol.GetAtoms()}
#        print(formal_charges)
#        images = Draw.MolsToImage([mol], subImgSize=(700,700))
#        mol = Chem.RemoveHs(mol,sanitize=False)
#        images.save('test.png')
#        exit()
        return mol

    @staticmethod
    def get_adj(mol):
        """get adjacency matrix. Returns nxn numpy array"""
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        np.fill_diagonal(adj, 1)
        return adj

    @staticmethod
    def get_bond_type(mol):
        """get bond order matrix. Returns nxn numpy array"""
        n_atoms = len(mol.GetAtoms())
        n_bonds = len(mol.GetBonds())
        bond_order = np.zeros([n_atoms, n_atoms])
        Chem.Kekulize(mol)
#        Chem.SanitizeMol(mol)
        for bond in range(n_bonds):
            i, j = mol.GetBondWithIdx(bond).GetBeginAtomIdx(), mol.GetBondWithIdx(bond).GetEndAtomIdx()
#            bond_order[i, j] = bond_order[j, i] = mol.GetBondWithIdx(bond).GetBondType()
            bond_order[i, j] = bond_order[j, i] = mol.GetBondWithIdx(bond).GetBondTypeAsDouble()
        return bond_order

    @staticmethod
    def get_formal_charges(mol):
        formal_charges = {atom.GetIdx(): atom.GetFormalCharge() for atom in mol.GetAtoms()}
        return formal_charges

    def __init__(self, f_input):
        """initialize class that gets features describing the structural information of a given molecule"""
        self.filename = f_input
        extension = f_input.split('.')[1]
        if extension == 'log':
            self.labels, self.coordinates = self.log_2_xyz()[1:]
        elif extension == 'xyz':
            self.labels, self.coordinates = self.get_xyz(f_input)
        else:
            print('ERROR. Please give file formatted as either xyz or Gaussian log.')
            exit()
        self.mol = self.get_mol(self.labels, self.coordinates)
        self.adj = self.get_adj(self.mol)
        self.bond_order = self.get_bond_type(self.mol)
        self.formal_charges = self.get_formal_charges(self.mol)


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    OBJ = MolCbh(FILENAME)
    print('MOL object created.')

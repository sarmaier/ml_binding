import sys
import os
from molmass import Formula

f_input = sys.argv[1]
pdb_name = f_input.split('_stripped_complex_ambpdb_08_28_23_ligand.xyz')[0]
coord_info = open(f_input, 'r').read().split('\n')[2:]
atoms = [[x.upper() for x in line.split(' ') if x][0] for line in coord_info if line]
atoms = [x.capitalize() for x in atoms]
mol_formula = ''.join(atoms)
molecular_obj = Formula(mol_formula)
mol_mass = molecular_obj.mass
if mol_mass > 900:
	print(pdb_name)
#	os.system('mv *' + pdb_name + '* mweight_900')

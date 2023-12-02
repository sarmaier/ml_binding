import os, sys, glob
from xyz2mol import xyz2mol
from xyz2mol import read_xyz_file 
from make_ligand_block import LigandBlock
import time as time
import signal

def handler(signum, frame):
	raise Exception("Action took too much time")


def get_mol(filename):                                                                                 
	name = filename.split("_stripped_complex_ambpdb_08_28_23_ligand.xyz")[0]
	signal.signal(signal.SIGALRM, handler)
	signal.alarm(10)
	try:
		a_num, charge, xyz_coord = read_xyz_file(filename)                                                 
		charged_fragments = True                                                                           
		quick = True                                                                                       
		huckel = False                                                                                     
		mol_object = xyz2mol(a_num, charge, xyz_coord, charged_fragments, quick, huckel)                  
		time.sleep(1) 
		print(name + ":file ok")
	except:
		mol_object = name + ":TIMEOUT"
		print(mol_object)
	signal.alarm(10)
	return mol_object                                                                                  

files = glob.glob("*ligand.xyz")
for filename in files:
	get_mol(filename)

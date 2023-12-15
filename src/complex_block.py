import os
import re
import numpy as np
from datetime import date
import time
import argparse


def parse_gbsa_block(str_, start_keyword, end_keyword):
    block = [x for x in re.findall(f"{start_keyword}(.*?){end_keyword}", str_, re.DOTALL)[0].split("\n") if x][2:]
    block = [[float(x) for x in y.split(" ") if "." in x][0] for y in block]
    return block[:4] + block[6:]


class ProcessingError(Exception):
    pass


class GbsaComplexBlock():
    def __init__(self, id):
        my_dir = os.getcwd()
        interaction_string = open(my_dir + f"/FINAL_RESULTS_MMGBSA_{id}.dat").read()
        delta = parse_gbsa_block(interaction_string, "Differences \(Complex - Receptor - Ligand\):", "\n\n------")
        ligand = parse_gbsa_block(interaction_string, "Ligand:", "\n\nDifference")
        receptor = parse_gbsa_block(interaction_string, "Receptor:", "\n\nLigand:")
        comp = parse_gbsa_block(interaction_string, "Complex:", "\n\nReceptor:")
        self.bind_properties = np.array(delta + ligand + receptor + comp).reshape((1, 20))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("PDB_ID", help="PDB ID of the molecule")
    args = parser.parse_args()

    pdb_id = args.PDB_ID
    if not os.path.isfile(f"FINAL_RESULTS_MMGBSA_{pdb_id}.dat"):
        print(f"Required files not found for PDB ID {pdb_id}")
    else:
        try:
            complex_gbsa = GbsaComplexBlock(pdb_id)
        except ProcessingError:
            print(f"Issue with processing {pdb_id}")
            exit()

    today = date.today()
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print("FINISHED PROCESSING")

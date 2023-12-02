import os
import re
import numpy as np
from datetime import date
import time
import argparse


def parse_gbsa_block(str_, start_keyword, end_keyword):
    block = [x for x in re.findall(f"{start_keyword}:(.*?){end_keyword}", str_, re.DOTALL)[0].split("\n") if x][2:]
    block = [[float(x) for x in y.split(" ") if "." in x][0] for y in block]
    block = block[:4] + block[6:]
    return block


class GbsaComplexBlock():
    def __init__(self, id):
        interaction_string = open(f"FINAL_RESULTS_MMGBSA_{id}.dat").read()
        delta = parse_gbsa_block(interaction_string, "Difference", "\n\n------")
        ligand = parse_gbsa_block(interaction_string, "Ligand", "Difference")
        receptor = parse_gbsa_block(interaction_string, "Receptor", "Ligand")
        comp = parse_gbsa_block(interaction_string, "Complex", "Receptor")
        self.bind_properties = np.array(delta + ligand + receptor + comp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("PDB_ID", help="PDB ID of the molecule")
    args = parser.parse_args()

    pdb_id = args.PDB_ID
    if not (os.path.isfile(f"FINAL_RESULTS_MMGBSA_{pdb_id}.dat") and
            os.path.isfile(f"FINAL_RESULTS_MMGBSA_per_residue_{pdb_id}_interaction.txt")):
        print(f"Required files not found for PDB ID {pdb_id}")
    else:
        try:
            complex_gbsa = GbsaComplexBlock(pdb_id)
        except Exception as e:
            print(f"Issue with processing PDB ID {pdb_id}: {e}")
            exit()

    today = date.today()
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print("FINISHED PROCESSING")
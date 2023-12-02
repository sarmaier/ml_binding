import os
import re
import numpy as np
import math
import traceback
from datetime import date
import time
import argparse


def gbsa_complex(str_):
    com = [x for x in re.findall(r"Complex:(.*?)Receptor:", str_, re.DOTALL)[0].split("\n") if x][2:]
    com = [[float(x) for x in y.split(" ") if "." in x][0] for y in com]
    com = com[:4] + com[6:]
    return com


def gbsa_receptor(str_):
    rec = [x for x in re.findall(r"Receptor:(.*?)Ligand:", str_, re.DOTALL)[0].split("\n") if x][2:]
    rec = [[float(x) for x in y.split(" ") if "." in x][0] for y in rec]
    rec = rec[:4] + rec[6:]
    return rec


def gbsa_ligand(str_):
    lig = [x for x in re.findall(r"Ligand:(.*?)Difference", str_, re.DOTALL)[0].split("\n") if x][2:]
    lig = [[float(x) for x in y.split(" ") if "." in x][0] for y in lig]
    lig = lig[:4] + lig[6:]
    return lig


def gbsa_delta(str_):
    delta = [x for x in re.findall(r"Difference(.*?)\n\n------", str_, re.DOTALL)[0].split("\n") if x][3:]
    delta = [[float(x) for x in y.split(" ") if "." in x][0] for y in delta]
    delta = delta[:4] + delta[6:]
    return delta


class GbsaComplexBlock():

    def __init__(self, id):
        interaction_string = open("FINAL_RESULTS_MMGBSA_" + id + ".dat").read()
        delta = gbsa_delta(interaction_string)
        ligand = gbsa_ligand(interaction_string)
        receptor = gbsa_receptor(interaction_string)
        comp = gbsa_complex(interaction_string)
        self.bind_properties = np.array(delta + ligand + receptor + comp)


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
            present_species.append(i)
        except AssertionError:
            pass
    for i in present_species:
        GbsaComplexBlock(i)
        try:
            complex_gbsa = GbsaComplexBlock(i)
        except Exception:
            print("Issue with FINAL_RESULTS_MMGBSA_" + i + ".dat")
            exit()
    today = date.today()
    time = time.strftime("%H:%M:%S", time.localtime())
    print("FINISHED PROCESSING")

import os
import re
import numpy as np
import math
import traceback
from datetime import date
import time
import pickle

def make_pickle(name, info):
    pickle_out = open(str(name) + '.pickle', 'wb')
    pickle.dump(info, pickle_out)
    pickle_out.close()
    return


def gbsa_complex(string):
    com = [x for x in re.findall(r"Complex:(.*?)Receptor:", string, re.DOTALL)[0].split("\n") if x][2:]
    com = [[float(x) for x in y.split(" ") if "." in x][0] for y in com]
    com = com[:4] + com[6:]
    return com


def gbsa_receptor(string):
    rec = [x for x in re.findall(r"Receptor:(.*?)Ligand:", string, re.DOTALL)[0].split("\n") if x][2:]
    rec = [[float(x) for x in y.split(" ") if "." in x][0] for y in rec]
    rec = rec[:4] + rec[6:]
    return rec


def gbsa_ligand(string):
    lig = [x for x in re.findall(r"Ligand:(.*?)Difference", string, re.DOTALL)[0].split("\n") if x][2:]
    lig = [[float(x) for x in y.split(" ") if "." in x][0] for y in lig]
    lig = lig[:4] + lig[6:]
    return lig


def gbsa_delta(string):
    delta = [x for x in re.findall(r"Difference(.*?)\n\n------", string, re.DOTALL)[0].split("\n") if x][3:]
    delta = [[float(x) for x in y.split(" ") if "." in x][0] for y in delta]
    delta = delta[:4] + delta[6:]
    return delta


if __name__ == '__main__':
    scales = {'f': 10 ** -15, 'p': 10 ** -12, 'n': 10 ** -9, 'u': 10 ** -6, 'm': 10 ** -3}
    try:
        assert os.path.isfile("INDEX_refined_data.2020")
        input_lines = open("INDEX_refined_data.2020", "r").read().split("\n")
        lines = [[x for x in y.split(" ") if x] for y in input_lines if "//" in y]
        ki = {x[0]: re.split("[a-z][A-Z]", x[4].split("=")[1])[0] for x in lines}
        ki_scale = {x[0]: re.split("([a-z][A-Z])", x[4].split("=")[1])[1].split("M")[0] for x in lines}
        exp_ki = {x: np.format_float_positional(float(ki[x]) * scales[ki_scale[x]], precision=4, fractional=False)
                  for x in ki}
        exp_pki = {x: np.format_float_positional(math.log10(float(exp_ki[x])), precision=4, fractional=False)
                   for x in exp_ki}
        species = [x for x in exp_pki]
    except AssertionError:
        print("INDEX_refined_data.2020 file needed to run.")
        exit()
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
        try:
            f_input = open("FINAL_RESULTS_MMGBSA_" + i + ".dat").read()
            delta = gbsa_delta(f_input)
            ligand = gbsa_ligand(f_input)
            receptor = gbsa_receptor(f_input)
            complex = gbsa_complex(f_input)
            mmgbsa_total_complex[i] = np.array(delta + ligand + receptor + complex)
        except Exception:
            print("Issue with FINAL_RESULTS_MMGBSA_" + i + ".dat")
            exit()
    today = date.today()
    time = time.strftime("%H:%M:%S", time.localtime())
    make_pickle("mmgbsa_total_complex_" + str(today) + "_" + str(time), mmgbsa_total_complex)
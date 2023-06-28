import sys
sys.path.append('/neurospin/lbi/monkeyfmri/deepstim/workspace/2023_ASM_tembedding/gitproject/2023_asm_tembedding/')

import os
import glob
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tembedding.eeg import eeg_markers

def main():
    os.environ["TEMBEDDING_DIR"] = "/neurospin/lbi/monkeyfmri/deepstim/database/DBS_database/"

    datadir = os.getenv("TEMBEDDING_DIR")
    if datadir is None:
        raise ValueError("Please specify the dataset directory in the "
                        "TEMBEDDING_DIR variable.")
    eegdir = os.path.join(datadir, "derivatives", "eeg_preproc")
    data = {"subject": [], "session": [], "condition": [], "path": []}
    for path in glob.glob(os.path.join(eegdir, "*", "*", "*.fif")):
        split = path.split(os.sep)
        data["condition"].append(split[-3])
        sid, ses = split[-2].split("_")[:2]
        data["subject"].append(sid)
        data["session"].append(ses)
        data["path"].append(path)
    data = pd.DataFrame.from_dict(data)
    print(data)
    print(data.groupby("subject").describe())

    outdir = "/neurospin/lbi/monkeyfmri/deepstim/workspace/2023_ASM_tembedding/temp"

    for d in data.iterrows():
        print(d[1]["session"])
        eeg_markers(d[1]['path'], outdir, basename = "{}_{}_{}".format(d[1]["subject"],d[1]["session"],d[1]["condition"]))

main()
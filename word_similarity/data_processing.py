"""
Extract four noun dissimilarity matrices from the Nishida et al. data.h5 file.

Matrices extracted (keys in HDF5):
- 'behav_noun_dissim' : behavior-derived noun dissimilarity
- 'wv_noun_dissim'    : word-vector-derived noun dissimilarity (1000-d fastText in the paper's notebook)
- 'mov1_noun_dissim'  : brain-derived noun dissimilarity (movie set 1; voxelwise models)
- 'mov2_noun_dissim'  : brain-derived noun dissimilarity (movie set 2; voxelwise models)

Also extracts the noun labels from 'nouns' for rows/columns.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from typing import Sequence
from scipy.spatial import distance as sdist
from scipy import stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
DATA = Path(ROOT / 'data')
sys.path.append(str(ROOT))

KEYS = {
    "behav": "behav_noun_dissim",
    "wv": "wv_noun_dissim",
    "mov1": "mov1_noun_dissim",
    "mov2": "mov2_noun_dissim",
    "labels": "nouns"
}

def read_required_mats():
    data_path = os.path.join(DATA, "data.h5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find HDF5 file: {data_path}")
    with h5py.File(data_path, "r") as f:
        missing = [k for k in KEYS.values() if k not in f]
        if missing:
            raise KeyError(f"Missing expected datasets in {data_path}: {missing}\n"
                           f"Found keys: {list(f.keys())}")
        behav = f[KEYS["behav"]][()]
        wv = f[KEYS["wv"]][()]
        mov1 = f[KEYS["mov1"]][()]
        mov2 = f[KEYS["mov2"]][()]
        labels = f[KEYS["labels"]].asstr()[()].tolist()
    return behav, wv, mov1, mov2, labels

def dissim_to_sim(matrix):
    dissim = matrix.astype(float).copy()
    max_val = np.nanmax(dissim)
    sim = max_val - dissim
    diag_val = np.nanmax(sim)
    np.fill_diagonal(sim, diag_val)
    return sim

def save_behav(behav, labels, sel_idx):
    behav = np.asarray(behav)
    S, P = behav.shape

    # per participant
    for s in range(S):
        condensed = behav[s, :]
        square = sdist.squareform(condensed)
        sim = dissim_to_sim(square)
        sim_select = sim[np.ix_(sel_idx, sel_idx)].astype(float)
        filename = f"noun_behav_subj{s:02d}.csv"
        outpath = os.path.join(DATA, "behav", filename)
        save_matrix(sim_select, labels, outpath)

    # overall mean
    mean_condensed = np.nanmean(behav, axis=0)
    mean_z = stats.zscore(mean_condensed, nan_policy="omit")
    mean_square = sdist.squareform(mean_z)
    mean_sim = dissim_to_sim(mean_square)
    mean_sim_select = mean_sim[np.ix_(sel_idx, sel_idx)].astype(float)
    filename = f"noun_behav_mean.csv"
    outpath = os.path.join(DATA, "behav", filename)
    save_matrix(mean_sim_select, labels, outpath)

def save_others(data, labels, name, algo_index: int = 0):
    # Each (3,5,1770) -> select algo, average seeds -> (1770,)
    condensed = np.nanmean(data[algo_index], axis=0)  # (1770,)
    square = sdist.squareform(condensed)
    sim = dissim_to_sim(square)
    filename = f"noun_{name}_mean.csv"
    outpath = os.path.join(DATA, f"{name}", filename)
    save_matrix(sim, labels, outpath)

def save_matrix(matrix: np.ndarray, labels: Sequence[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(path, encoding="utf-8")

def pick_subset_indices(seed: int = 1):
    rng = np.random.default_rng(seed)
    grp1 = rng.choice(np.arange(1, 11) - 1, size=5, replace=False)
    grp2 = rng.choice(np.arange(11, 21) - 1, size=5, replace=False)
    grp3 = rng.choice(np.arange(21, 31) - 1, size=5, replace=False)
    grp4 = rng.choice(np.arange(31, 41) - 1, size=5, replace=False)
    grp5 = rng.choice(np.arange(51, 61) - 1, size=5, replace=False)
    sel = np.concatenate([np.sort(grp1), np.sort(grp2), np.sort(grp3), np.sort(grp4), np.sort(grp5)])
    return sel.astype(int).tolist()

def main():
    behav, wv, mov1, mov2, labels = read_required_mats()
    sel_idx = pick_subset_indices()
    selected_labels = [labels[i] for i in sel_idx]

    save_behav(behav, selected_labels, sel_idx)
    # save_others(wv, labels, "wv")
    # save_others(mov1, labels, "mov1")
    # save_others(mov2, labels, "mov2")

    labels_path = os.path.join(DATA, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as fp:
        for w in selected_labels:
            fp.write(w + "\n")

if __name__ == "__main__":
    main()
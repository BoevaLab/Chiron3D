import numpy as np
import os
import re

base_dir = "/cluster/work/boeva/shoenig/ews-ml/prelim_results/loop_calling/TC71_WT/BORZOI_FULLLORA"

for fname in os.listdir(base_dir):
    if not fname.endswith(".npy"):
        continue

    fpath = os.path.join(base_dir, fname)

    # Case 1: sym files (X,Y)
    if "X,Y" in fname:
        # Rename left→anchor, right→non_anchor, X,Y→sym
        new_name = fname.replace("X,Y", "sym")
        new_name = new_name.replace("_left", "_anchor").replace("_right", "_non_anchor")
        new_path = os.path.join(base_dir, new_name)
        print(f"Renaming {fname} -> {new_name}")
        arr = np.load(fpath, mmap_mode="r")  # no modification, just copy
        np.save(new_path, arr)

    # Case 2: asym files (X or Y)
    elif "X_Method" in fname and ("scores" in fname or "sequences" in fname):
        if "_left" in fname:
            partner = fname.replace("X_Method", "Y_Method").replace("_left", "_right")
            if os.path.exists(os.path.join(base_dir, partner)):
                arr1 = np.load(fpath)
                arr2 = np.load(os.path.join(base_dir, partner))
                merged = np.concatenate([arr1, arr2], axis=0)
                new_name = fname.replace("X_Method", "asym_Method").replace("_left", "_anchor")
                print(f"Merging {fname} + {os.path.basename(partner)} -> {new_name}")
                np.save(os.path.join(base_dir, new_name), merged)

        elif "_right" in fname:
            partner = fname.replace("X_Method", "Y_Method").replace("_right", "_left")
            if os.path.exists(os.path.join(base_dir, partner)):
                arr1 = np.load(fpath)
                arr2 = np.load(os.path.join(base_dir, partner))
                merged = np.concatenate([arr1, arr2], axis=0)
                new_name = fname.replace("X_Method", "asym_Method").replace("_right", "_non_anchor")
                print(f"Merging {fname} + {os.path.basename(partner)} -> {new_name}")
                np.save(os.path.join(base_dir, new_name), merged)

import numpy as np
import glob
import re
import os
import sys

STRIPE = os.environ.get("STRIPE", "X,Y") 

def combine_files(pattern, output_file, axis=0):
    # Find files matching the pattern
    files = glob.glob(pattern)
    # Sort files by the numeric prefix before the first underscore
    files = sorted(files, key=lambda x: int(re.match(r"(\d+)_", x).group(1)))
    arrays = []
    for f in files:
        print(f"Loading {f}")
        arr = np.load(f)
        arrays.append(arr)
    # Concatenate arrays along the given axis (default is 0)
    combined = np.concatenate(arrays, axis=axis)
    # Save the combined array
    np.save(output_file, combined)
    print(f"[OK] Combined {len(files)} files into {output_file}")

    for f in files:
        try:
            os.remove(f)
            print(f"Deleted source file: {f}")
        except OSError as e:
            print(f"Error deleting {f}: {e}")


def detect_method():
    # only look at numbered files
    sample = glob.glob(f"[0-9]*_Imp_Stripe_{STRIPE}_Method_*.npy")
    if not sample:
        print("No numbered files found; aborting.", file=sys.stderr)
        sys.exit(1)

    # extract whatever’s between “Method_” and the next “_…”
    m = re.match(
        rf"^\d+_Imp_Stripe_{re.escape(STRIPE)}_Method_"
        r"(.*?)_"  # capture the method name
        r"(scores_left|scores_right|sequences_left|sequences_right)\.npy",
        os.path.basename(sample[0])
    )
    if not m:
        print("Couldn't parse method; aborting.", file=sys.stderr)
        sys.exit(1)

    return m.group(1)


if __name__ == "__main__":
    method = detect_method()
    print(f"Detected method: {method}")

    file_types = [
        "scores_left",
        "scores_right",
        "sequences_left",
        "sequences_right",
    ]

    for t in file_types:
        # only numbered sources
        pattern = f"[0-9]*_Imp_Stripe_{STRIPE}_Method_{method}_{t}.npy"
        # cleaned-up output name
        output_fn = f"Imp_Stripe_{STRIPE}_Method_{method}_{t}.npy"
        combine_files(pattern, output_fn, axis=0)

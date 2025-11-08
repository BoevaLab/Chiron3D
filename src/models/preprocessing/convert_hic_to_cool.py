#!/usr/bin/env python
import sys
from hic2cool import hic2cool_convert


def main():
    if len(sys.argv) != 4:
        print("Usage: {} <input.hic> <output.cool> <resolution>".format(sys.argv[0]))
        sys.exit(1)
    hic_file = sys.argv[1]
    cool_file = sys.argv[2]
    resolution = int(sys.argv[3])

    # Optionally: Ensure the output directory exists:
    import os
    out_dir = os.path.dirname(cool_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    hic2cool_convert(hic_file, cool_file, resolution)


if __name__ == '__main__':
    main()

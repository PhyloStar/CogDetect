#!/usr/bin/env python3

import sys
import argparse

import lingpy
import pandas
import pickle

from online_pmi import clean_word, ipa2asjp

def main():
    """Run the cli."""
    parser = argparse.ArgumentParser(description="Align forms in cognate classes")
    parser.add_argument(
        "input", nargs="?",
        type=argparse.FileType('r'),
        default=sys.stdin,
        help="A LingPy TSV file to read")
    parser.add_argument(
        "output", nargs="?",
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="Filename to write LingPy TSV output to")
    parser.add_argument(
        "--scoredict",
        type=argparse.FileType('rb'),
        default=False,
        help="Read a PMI Score Match dictionary from this pickle file")
    parser.add_argument(
        "--alignment-column",
        default="ALIGNMENT",
        help="Name of the new automatic alignment column in the output")
    parser.add_argument(
        "--cogid-column",
        default="COGID",
        help="Name of the column containing the cognate classes to use")
    args = parser.parse_args()
    
    data = pandas.read_csv(
        args.input, sep="\t")

    scoredict = pickle.load(args.scoredict)
    
    data[args.alignment_column] = ""
    for cogid, block in data.groupby(args.cogid_column):
        try:
            forms = [clean_word(w) for w in block["ASJP"]]
        except KeyError:
            forms = [ipa2asjp(w) for w in block["IPA"]]
        aligned = lingpy.align.multiple.mult_align(
            forms,
            gop=-2.0,
            scale=0.8,
            scoredict=scoredict,
            tree_calc='neighbor')
        data[
            data[args.cogid_column] == cogid][
                args.alignment_column] = aligned
        print(aligned)

if __name__ == "__main__":
    main()

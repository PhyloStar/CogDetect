#!/usr/bin/env python3

"""Align words based on stepwise EM alignments with PMI scores."""

import itertools as it
import collections

import sys
import igraph, utils
import numpy as np
import random, codecs
import infomapcog.clustering as clust
import infomapcog.distances as distances

import argparse

import csv
import pickle

import lingpy

import infomapcog.ipa2asjp as ipa2asjp
from infomapcog.dataio import (read_data_cldf, read_data_lingpy,
                               read_data_ielex_type, multi_align,
                               MaxPairDict)

import newick
readers = {
    "ielex": read_data_ielex_type,
    "cldf": read_data_cldf,
    "lingpy": read_data_lingpy,
    }

if __name__ == "__main__":

    # TODO:
    # - Add a ML based estimation of distance or a JC model for distance
    #   between two sequences
    # - Separate clustering code.
    # - Add doculect distance as regularization

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        "--guide-tree",
        type=argparse.FileType("r"),
        help="""A Newick file containing a single guide tree to combine
        multiple alignments. (Separate guide trees for different families are
        not supported yet.)""")
    parser.add_argument(
        "data",
        type=argparse.FileType("r"),
        help="IELex-style data file to read")
    parser.add_argument(
        "--transcription",
        default='ASJP',
        help="""The transcription convention (IPA, ASJP, …) used in the data
        file""")
    parser.add_argument(
        "--pmidict",
        type=argparse.FileType("rb"),
        help="Read PMI dictionary from this (pickle) file.")
    parser.add_argument(
        "--reader",
        choices=list(readers.keys()),
        default="ielex",
        help="Data file format")

    args = parser.parse_args()

    data_dict, cogid_dict, words_dict, langs_list, char_list = (
        readers[args.reader](args.data, data=args.transcription))
    print("Character list:", char_list, "({:d})".format(len(char_list)))

    if args.pmidict:
        pmidict = pickle.load(args.pmidict)

    correspondences = collections.defaultdict(list)
    if args.guide_tree:
        tree = newick.load(args.guide_tree)[0]

    for group, (languages, concepts, alignment) in multi_align(
            cogid_dict, tree,
            lodict=MaxPairDict(pmidict),
            gop=None, gep=-1.75).items():
        if len(languages) > 1:
            print(languages)
            print(concepts)
            print(alignment)

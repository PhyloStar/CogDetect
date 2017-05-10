#!/usr/bin/python

import itertools

import newick
import pandas

import sys
import argparse

parser = argparse.ArgumentParser(
    description="Read a LingPy or CLDF file with alignments and list sound correspondences in order of frequency")
parser.add_argument(
    "data",
    type=argparse.FileType('r'),
    help="TSV file to read")
parser.add_argument(
    "--quiet",
    action='store_true',
    help="Suppress output of identical correspondences")
parser.add_argument(
    "--all-languages",
    action='store_true',
    help="Find correspondences betwenn all languages, instead of between pairs")
parser.add_argument(
    "--tree",
    type=argparse.FileType('r'),
    help="A reference tree, also used to restrict the languages shown")

args = parser.parse_args()

data = pandas.read_csv(args.data, sep="\t")

if 'DOCULECT' in data.columns:
    LANG = 'DOCULECT_ID'
    CONCEPT = 'CONCEPT'
    SIMID = 'COGID'
    ALIGNMENT = 'ALIGNMENT'
elif 'Language_ID' in data.columns:
    LANG = 'Language_ID'
    CONCEPT = 'Concept'
    SIMID = 'Cognate Set'
    ALIGNMENT = 'Alignment'
else:
    raise ValueError('Unrecognized column names')
    
if args.tree:
    tree = newick.load(args.tree)[0]
    languages = tree.get_leaf_names()
else:
    languages = data[LANG].unique()

correspondences = {}

if args.all_languages:
    for simid, sims in data.groupby(SIMID):
        alignments = sims[[CONCEPT, LANG, ALIGNMENT]].set_index(LANG)
        try:
            by_lang = alignments.loc[languages]
        except KeyError:
            continue
        code = by_lang[CONCEPT].unique()[0]
        if pandas.isnull(code):
            code = by_lang[CONCEPT].unique()[1]
        i = 0
        while True:
            correspondence = by_lang[ALIGNMENT].str[i]
            if pandas.isnull(correspondence).all():
                break
            if sum(~pandas.isnull(correspondence)) == 1:
                continue
            correspondences.setdefault(
                tuple(correspondence), []).append(
                    (code, i))
            i += 1
else:
    for simid, sims in data.groupby(SIMID):
        relevant_colums = sims[[CONCEPT, LANG, ALIGNMENT]]
        for (i1, d1), (i2, d2) in itertools.combinations(
                relevant_colums.iterrows(), 2):
            if d1[1] < d2[1]:
                d1, d2 = d2, d1
            c1, l1, a1 = d1
            a1 = a1.split()

            c2, l2, a2 = d2
            a2 = a2.split()

            if l1 not in languages or l2 not in languages:
                continue

            if len(a1) != len(a2):
                print("Alignments {:} and {:} don't match!".format(
                    tuple(d1), tuple(d2)), file=sys.stderr)
                continue

            for s1, s2 in zip(a1, a2):
                if s1 in {'', '-'} or s2 in {'', '-'}:
                    continue
                if s1 == s2 and args.quiet:
                    continue
                correspondences.setdefault(
                    (l1, s1, l2, s2), []).append((c1, a1, c2, a2))

for key, val in sorted(correspondences.items(),
                       key=lambda x: len(x[1])):
    print(key, len(val), val)

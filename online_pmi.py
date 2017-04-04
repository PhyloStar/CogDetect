#!/usr/bin/env python3

from collections import defaultdict
import itertools as it
import sys, distances, igraph, utils
import numpy as np
import random, codecs
import Clustering as clust

import argparse

import csv

def clean_word(w):
    w = w.replace("-","")
    w = w.replace(" ", "")
    w = w.replace("%","")
    w = w.replace("~","")
    w = w.replace("*","")
    w = w.replace("$","")
    w = w.replace("\"","")
    w = w.replace("K","k")
    w = w.replace("|","")
    w = w.replace(".","")
    w = w.replace("+","")
    w = w.replace("·","")
    w = w.replace("?","")
    w = w.replace("’","")
    w = w.replace("]","")
    w = w.replace("[","")
    w = w.replace("=","")
    w = w.replace("_","")
    w = w.replace("<","")
    w = w.replace(">","")
    w = w.replace("‐","")
    w = w.replace("ᶢ","")
    w = w.replace("C","c")
    w = w.replace("L","l")
    w = w.replace("W","w")
    w = w.replace("T","t")
    return w


def ipa2sca(w):
    return "".join(tokens2class(ipa2tokens(w), 'asjp')).replace("0","")


def read_data_ielex_type(datafile):
    """Read an IELex style TSV file."""
    line_id = 0
    data_dict = defaultdict(lambda : defaultdict())
    cogid_dict = defaultdict(lambda : defaultdict())
    words_dict = defaultdict(lambda : defaultdict(list))
    langs_list = []
    
    datafile.readline()
    for line in datafile:
        line = line.strip()
        arr = line.split("\t")
        lang = arr[0]

        concept = arr[2]
        cogid = arr[6]
        cogid = cogid.replace("-","")
        cogid = cogid.replace("?","")
        asjp_word = clean_word(arr[5].split(",")[0])

        for ch in asjp_word:
            if ch not in char_list:
                char_list.append(ch)

        if len(asjp_word) < 1:
            continue

        data_dict[concept][line_id,lang] = asjp_word
        cogid_dict[concept][line_id,lang] = cogid
        words_dict[concept][lang].append(asjp_word)
        if lang not in langs_list:
            langs_list.append(lang)
        line_id += 1
    
    print(list(data_dict.keys()))
    return (data_dict, cogid_dict, words_dict, langs_list)


def read_data_cldf(datafile, sep="\t", char_list=set()):
    """Read a CLDF file in TSV or CSV format."""
    reader = csv.DictReader(
        datafile,
        dialect='excel' if sep == ',' else 'excel-tab')
    langs = set()
    for line, row in enumerate(reader):
        lang = row["Language ID"]
        langs.add(lang)

        asjp_word = clean_word(row["ASJP"])
        if not asjp_word:
            continue

        for ch in asjp_word:
            if ch not in char_list:
                char_list.append(ch)

        concept = row["Feature ID"]
        cogid = row["Cognate Class"]

        data_dict[concept][line, lang] = asjp_word
        cogid_dict[concept][line, lang] = cogid
        words_dict[concept].setdefault(lang, []).append(asjp_word)

    print(list(data_dict.keys()))
    return (data_dict, cogid_dict, words_dict, list(langs))


def read_data_lingpy(datafile, sep="\t", char_list=set()):
    """Read a Lingpy file in TSV or CSV format."""
    reader = csv.DictReader(
        datafile,
        dialect='excel' if sep == ',' else 'excel-tab')
    langs = set()
    for line, row in enumerate(reader):
        lang = row["DOCULECT"]
        langs.add(lang)

        asjp_word = clean_word(row["ASJP"])
        if not asjp_word:
            continue

        for ch in asjp_word:
            if ch not in char_list:
                char_list.append(ch)

        concept = row["CONCEPT"]
        cogid = row["COGID"]

        data_dict[concept][line, lang] = asjp_word
        cogid_dict[concept][line, lang] = cogid
        words_dict[concept].setdefault(lang, []).append(asjp_word)

    print(list(data_dict.keys()))
    return (data_dict, cogid_dict, words_dict, list(langs))


def calc_pmi(alignment_dict, char_list, scores, initialize=False):
    sound_dict = defaultdict(float)
    relative_align_freq = 0.0
    relative_sound_freq = 0.0
    count_dict = defaultdict(float)
    
    if initialize == True:
        for c1, c2 in it.product(char_list, repeat=2):
            if c1 == "-" or c2 == "-":
                continue
            count_dict[c1,c2] += 0.001
            count_dict[c2,c1] += 0.001
            sound_dict[c1] += 0.001
            sound_dict[c2] += 0.001
            relative_align_freq += 0.001
            relative_sound_freq += 0.002

    for alignment, score in zip(alignment_dict, scores):
        score = 1.0
        for a1, a2 in alignment:
            if a1 == "-" or a2 == "-":
                continue
            count_dict[a1,a2] += 1.0*score
            count_dict[a2,a1] += 1.0*score
            sound_dict[a1] += 2.0*score
            sound_dict[a2] += 2.0*score
            #relative_align_freq += 2.0
            #relative_sound_freq += 2.0

    relative_align_freq = sum(list(count_dict.values()))
    relative_sound_freq = sum(list(sound_dict.values()))
    
    for a in count_dict.keys():
        m = count_dict[a]
        if m <=0: print(a, m)
        assert m>0

        num = np.log(m)-np.log(relative_align_freq)
        denom = np.log(sound_dict[a[0]])+np.log(sound_dict[a[1]])-(2.0*np.log(relative_sound_freq))
        val = num - denom
        count_dict[a] = val
    
    return count_dict


readers = {
    "ielex": read_data_ielex_type,
    "cldf": read_data_cldf,
    "lingpy": read_data_lingpy,
    }

if __name__ == "__main__":

    ##TODO: Add a ML based estimation of distance or a JC model for distance between two sequences
    ##Separate clustering code.
    ##Add doculect distance as regularization

    parser = argparse.ArgumentParser(
        description="Calculate pmi scores, or something.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="Maximum number of iterations")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="TODO tolerance")
    parser.add_argument(
        "--infomap-threshold",
        type=float,
        default = 0.5,
        help="Threshold for the INFOMAP algorithm")
    parser.add_argument(
        "--min-batch",
        type=int,
        default=256,
        help="TODO min batch")
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="TODO margin")
    parser.add_argument(
        "data",
        type=argparse.FileType("r"),
        help="IELex-style data file to read")
    parser.add_argument(
        "--reader",
        choices=list(readers.keys()),
        help="Data file format")
    args = parser.parse_args()
                        
    random.seed(args.seed)
    MAX_ITER = args.max_iter
    tolerance = args.tolerance
    infomap_threshold = args.infomap_threshold
    min_batch = args.min_batch
    margin = args.margin

    dataname = args.data

    char_list = []

    data_dict, cogid_dict, words_dict, langs_list = (
        readers[args.reader](dataname))
    print("Character list \n\n", char_list)
    print("Length of character list ", len(char_list))

    word_list = []

    for concept in data_dict:
        print(concept)
        words = []
        for idx in data_dict[concept]:
            words.append(data_dict[concept][idx])
        for x, y in it.combinations(words, r=2):
            #if distances.nw(x, y, lodict=None, gp1=-2.5,gp2=-1.75)[0] > 0.0:
            if distances.ldn(x, y) <=0.5:
                word_list += [[x,y]]

    #word_list = [line.strip().split()[0:2] for line in open(fname).readlines()]
    #char_list = [line.strip() for line in open("sounds41.txt").readlines()]


    pmidict = None
    n_examples, n_updates, alpha = len(word_list), 0, 0.75
    n_wl = len(word_list)
    print("Size of initial list ", n_wl)

    #for n_iter in range(MAX_ITER):
    #    print("Iteration ", n_iter)
    #    algn_list, scores, pruned_wl = [], [], []
    #    n_zero = 0.0
    #    for w1, w2 in word_list:
    #        if n_iter == 0:
    #            s, alg = distances.nw(w1, w2, lodict=pmidict, gp1=-1.0,gp2=-0.5)
    #        else:
    #            s, alg = distances.nw(w1, w2, lodict=pmidict)
    #        s = s/max(len(w1), len(w2))
    #        if s <=0.0:
    #            n_zero += 1.0
    #            continue
    #        algn_list.append(alg)
    #        scores.append(s)
    #        pruned_wl.append([w1, w2])
    #    word_list = pruned_wl[:]
    #    n_wl = len(word_list)
    #    pmidict = calc_pmi(algn_list, char_list, scores, initialize=True)
    #    print("Non zero examples ", n_wl, n_wl-n_zero)

    #bin_mat = infomap_concept_evaluate_scores(data_dict, pmidict, -2.5, -1.75)
    #sys.exit(1)
    #for k, v in pmidict.items():
    #    print(k, v)


    pmidict = defaultdict(float)

    for n_iter in range(MAX_ITER):
        random.shuffle(word_list)
        pruned_wl = []
        n_zero = 0.0
        print("Iteration ", n_iter)
        for idx in range(0, n_wl, min_batch):
            wl = word_list[idx:idx+min_batch]
            eta = np.power(n_updates+2, -alpha)
            algn_list, scores = [], []
            for w1, w2 in wl:
                #print(w1,w2,sc)
                if not pmidict:
                    s, alg = distances.nw(w1, w2, lodict=None, gp1=-2.5, gp2=-1.75)
                else:
                    s, alg = distances.nw(w1, w2, lodict=pmidict, gp1=-2.5, gp2=-1.75)
                if s <= margin:
                    n_zero += 1.0
                    continue
                #s = s/max(len(w1), len(w2))
                algn_list.append(alg)
                scores.append(s)
                #pruned_wl.append([w1[::-1], w2[::-1], s])	  
                pruned_wl.append([w1, w2])	  
            mb_pmi_dict = calc_pmi(algn_list, char_list, scores, initialize=True)
            for k, v in mb_pmi_dict.items():
                pmidict_val = pmidict[k]
                pmidict[k] = (eta*v) + ((1.0-eta)*pmidict_val)
            n_updates += 1
        print("Non zero examples ", n_wl, n_wl-n_zero, " number of updates ", n_updates)
        word_list = pruned_wl[:]
        n_wl = len(word_list)
        #infomap_concept_evaluate_scores(data_dict, pmidict, -2.5, -1.75)

    #for k, v in pmidict.items():
    #    print(k, v)
    bin_mat = clust.infomap_concept_evaluate_scores(data_dict, pmidict, -2.5, -1.75, infomap_threshold, cogid_dict)
    #nchar, nlangs = np.array(bin_mat).shape

    sys.exit(1)
    print("begin data;")
    print("   dimensions ntax=", str(nlangs), "nchar=", str(nchar), ";\nformat datatype=restriction interleave=no missing= ? gap=-;\nmatrix\n")

    for row, lang in zip(np.array(bin_mat).T, lang_list):
        #print(row,len(row), "\n")
        rowx = "".join([str(x) for x in row])
        print(lang, "\t", rowx.replace("2","?"))
    print(";\nend;")

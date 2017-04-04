#!/usr/bin/env python3

import os

from collections import defaultdict
import codecs
import glob
from lingpy import ipa2tokens, tokens2class


def read_ipa_to_asjp(
        filename=os.path.join(os.path.dirname(__file__),
                              "ipa2asjp.txt")):
    """Read a SSV mapping IPA symbols to ASJP classes."""
    ipa_to_asjp = {}

    f = codecs.open(filename, "r", encoding="utf-8")

    for line in f:
        line = line.replace("\n", "")
        line = line.replace("\r", "")
        ipa, asjp = line.split(" ")
        ipa_to_asjp[ipa] = asjp
    f.close()
    return ipa_to_asjp


ipa_to_asjp = read_ipa_to_asjp()


def ipa2asjp(ipa):
    """Convert an IPA string into a ASJP token string."""
    tokenized_word = ipa2tokens(ipa)
    asjp_list = [''.join([i for i in ipa_to_asjp.get(x, tokens2class(x, 'asjp'))])
                 for x in tokenized_word]
    print(''.join(asjp_list))
    return ''.join(asjp_list)


def read_convert_ipa_asjp():
    f_trace = codecs.open("test_ipa2asjp.txt", "w", encoding="utf-8")
    for file_name in glob.iglob("data/*tsv"):
        f_trace.write(file_name+"\n")
        f = codecs.open(file_name, "r", encoding="utf-8")
        fout = codecs.open(file_name+".asjp", "w", encoding="utf-8")
        header = f.readline()
        fout.write(header)
        header = header.split("\t")
        for line in f:
            arr = line.split("\t")
            arr[5] = ipa2asjp(arr[5])
            fout.write(
                "\t".join(arr))
        f.close()
        fout.close()
    f_trace.close()



if __name__ == "__main__":
    read_convert_ipa_asjp()

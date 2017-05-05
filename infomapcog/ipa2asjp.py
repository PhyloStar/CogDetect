#!/usr/bin/env python3

import os

import itertools

import codecs
import glob
from lingpy import ipa2tokens, tokens2class


def read_ipa_to_asjp(
        filename=os.path.join(os.path.dirname(__file__), "../data",
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


def tokenize_word_reversibly(ipa):
    """Reversibly convert an IPA string into a list of tokens.

    In contrast to LingPy's tokenize_word, do this without removing
    symbols. This means that the original IPA string can be recovered
    from the tokens.

    """
    tokenized_word = ipa2tokens(ipa, merge_vowels=False)
    token = 0
    index = 0
    for i in ipa:
        try:
            tokenized_word[token][index]
        except IndexError:
            token += 1
            index = 0
        try:
            if i != tokenized_word[token][index]:
                if index == 0:
                    tokenized_word.insert(token, i)
                else:
                    tokenized_word[token] = (
                        tokenized_word[token][:index] +
                        i +
                        tokenized_word[token][index:])
        except IndexError:
            tokenized_word.append(i)
        index += 1
    assert ''.join(tokenized_word) == ipa
    return tokenized_word


def ipa2asjp(ipa):
    """Convert an IPA string into a ASJP token string.

    This function tries to preserve the len of the token string.

    """
    asjp_list = [t for x in tokenize_word_reversibly(ipa)
                 for t, char in itertools.zip_longest(
                         tokens2class(x, 'asjp'),
                         "0")]
    assert len(''.join(asjp_list)) == len(ipa)
    return ''.join(asjp_list)


def ipa2sca(ipa):
    """Convert an IPA string into a SCA token string.

    This function tries to preserve the len of the token string.

    """
    sca_list = [t for x in tokenize_word_reversibly(ipa)
                 for t, char in itertools.zip_longest(
                         tokens2class(x, 'sca'),
                         "0")]
    assert len(''.join(sca_list)) == len(ipa)
    return ''.join(sca_list)


def read_convert_ipa_asjp():
    """Convert IPA IELex files to ASJP."""
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

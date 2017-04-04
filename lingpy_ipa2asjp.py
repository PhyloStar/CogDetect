#!/usr/bin/env python3

import os

from collections import defaultdict
import codecs
import glob
from lingpy import ipa2tokens


def read_convert_ipa_asjp():
    ipa2asjp_dict = defaultdict(str)
    f = codecs.open(os.path.join(os.path.dirname(__file__),
                                 "ipa2asjp.txt"),
                    "r", encoding="utf-8")
    for line in f:
        line = line.replace("\n", "")
        line = line.replace("\r", "")
        temp = line.split(" ")
        ipa = temp[0]
        asjp = ""
        if temp[1] == "":
            asjp = "0"
        else:
            asjp = temp[1]
        ipa2asjp_dict[ipa] = asjp
    f.close()

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
            lang, global_id, ipa_word, cog_class = arr[0], arr[3], arr[5], arr[6]
            tokenized_word = ipa2tokens(ipa_word)
            asjp_list = [ipa2asjp_dict[x] for x in ipa_word]
            asjp_word = ""
            for k in asjp_list:
                k = k.replace("0", "")
                if k == "0":
                    f_trace.write(ipa_word+"\t" + "".join(asjp_list)+"\n")
                    continue
                else:
                    asjp_word += k
            arr[5] = asjp_word
            fout.write(
                "\t".join(arr))
        f.close()
        fout.close()
    f_trace.close()
    

if __name__ == "__main__":
    read_convert_ipa_asjp()

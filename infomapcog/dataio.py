"""Data manipulation tools for cognate detection."""

import numpy as np
import itertools as it
import collections

import csv

import lingpy

from . import distances
from . import ipa2asjp


def clean_word(w):
    """Clean a string to reduce non-IPA noise."""
    w = w.replace("-", "")
    w = w.replace(" ", "")
    w = w.replace("%", "")
    w = w.replace("~", "")
    w = w.replace("*", "")
    w = w.replace("$", "")
    w = w.replace("\"", "")
    w = w.replace("|", "")
    w = w.replace(".", "")
    w = w.replace("+", "")
    w = w.replace("·", "")
    w = w.replace("?", "")
    w = w.replace("’", "")
    w = w.replace("]", "")
    w = w.replace("[", "")
    w = w.replace("=", "")
    w = w.replace("_", "")
    w = w.replace("<", "")
    w = w.replace(">", "")
    w = w.replace("‐", "")
    w = w.replace("ᶢ", "")
    w = w.replace("C", "c")
    w = w.replace("K", "k")
    w = w.replace("L", "l")
    w = w.replace("W", "w")
    w = w.replace("T", "t")
    w = w.replace('dʒ͡', 'd͡ʒ')
    w = w.replace('ʤ', 'd͡ʒ')
    w = w.replace('Ɂ', 'Ɂ')
    return w


def read_data_ielex_type(datafile, char_list=set(),
                         cogids_are_cross_semantically_unique=False,
                         data='ASJP'):
    """Read an IELex style TSV file."""
    line_id = 0
    data_dict = collections.defaultdict(lambda: collections.defaultdict())
    cogid_dict = {}
    words_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    langs_list = []

    # Ignore the header line of the data file.
    datafile.readline()
    for line in datafile:
        line = line.strip()
        arr = line.split("\t")
        lang = arr[0]

        concept = arr[2]
        cogid = arr[6]
        cogid = cogid.replace("-", "")
        cogid = cogid.replace("?", "")
        if data == 'ASJP':
            asjp_word = clean_word(arr[5].split(", ")[0])
        else:
            raise NotImplementedError

        for ch in asjp_word:
            if ch not in char_list:
                char_list.add(ch)

        if len(asjp_word) < 1:
            continue

        data_dict[concept][line_id, lang] = asjp_word
        cogid_dict.setdefault(cogid
                              if cogids_are_cross_semantically_unique
                              else (cogid, concept), set()).add(
            (lang, concept, asjp_word))
        words_dict[concept][lang].append(asjp_word)
        if lang not in langs_list:
            langs_list.append(lang)
        line_id += 1

    return (data_dict,
            list(cogid_dict.values()),
            words_dict,
            langs_list,
            char_list)


def read_data_cldf(datafile, sep="\t", char_list=set(),
                   cogids_are_cross_semantically_unique=True,
                   data='ASJP'):
    """Read a CLDF file in TSV or CSV format."""
    reader = csv.DictReader(
        datafile,
        dialect='excel' if sep == ', ' else 'excel-tab')
    langs = set()
    data_dict = collections.defaultdict(lambda: collections.defaultdict())
    cogid_dict = collections.defaultdict(lambda: collections.defaultdict())
    words_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for line, row in enumerate(reader):
        lang = row["Language ID"]
        langs.add(lang)

        if data == 'ASJP':
            try:
                asjp_word = clean_word(row["ASJP"])
            except KeyError:
                asjp_word = ipa2asjp.ipa2asjp(row["IPA"])
        elif data == 'IPA':
            asjp_word = tuple(
                lingpy.ipa2tokens(row["IPA"], merge_vowels=False))
        else:
            asjp_word = row[data]

        if not asjp_word:
            continue

        for ch in asjp_word:
            if ch not in char_list:
                char_list.add(ch)

        concept = row["Feature ID"]
        cogid = row["Cognate Class"]

        data_dict[concept][line, lang] = asjp_word
        cogid_dict.setdefault(cogid
                              if cogids_are_cross_semantically_unique
                              else (cogid, concept), set()).add(
            (lang, concept, asjp_word))
        words_dict[concept].setdefault(lang, []).append(asjp_word)

    return (data_dict,
            list(cogid_dict.values()),
            words_dict,
            list(langs),
            char_list)


def read_data_lingpy(datafile, sep="\t", char_list=set(),
                     cogids_are_cross_semantically_unique=True,
                     data='ASJP'):
    """Read a Lingpy file in TSV or CSV format."""
    reader = csv.DictReader(
        datafile,
        dialect='excel' if sep == ', ' else 'excel-tab')
    langs = set()
    data_dict = collections.defaultdict(collections.defaultdict)
    cogid_dict = {}
    words_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for line, row in enumerate(reader):
        lang = row.get("DOCULECT_ID", row["DOCULECT"])
        langs.add(lang)

        if data == 'ASJP':
            try:
                asjp_word = clean_word(row["ASJP"])
            except KeyError:
                asjp_word = ipa2asjp.ipa2asjp(row["IPA"])
        elif data == 'IPA':
            asjp_word = tuple(ipa2asjp.tokenize_word_reversibly(
                clean_word(row["IPA"])))
        else:
            asjp_word = row[data]

        if not asjp_word:
            continue

        for ch in asjp_word:
            if ch not in char_list:
                char_list.add(ch)

        concept = row["CONCEPT"]
        cogid = row["COGID"]

        data_dict[concept][line, lang] = asjp_word
        cogid_dict.setdefault(cogid
                              if cogids_are_cross_semantically_unique
                              else (cogid, concept), set()).add(
            (lang, concept, asjp_word))
        words_dict[concept].setdefault(lang, []).append(asjp_word)

    return (data_dict,
            list(cogid_dict.values()),
            words_dict,
            list(langs),
            char_list)


def calc_pmi(alignments, scores=None):
    """Calculate a pointwise mutual information dictionary from alignments.

    Given a sequence of pairwaise alignments and their relative
    weights, calculate the logarithmic pairwise mutual information
    encoded for the character pairs in the alignments.

    """
    if scores is None:
        scores = it.cycle([1])

    sound_dict = collections.defaultdict(float)
    count_dict = collections.defaultdict(float)

    for alignment, score in zip(alignments, scores):
        for a1, a2 in alignment:
            if a1 == "" or a2 == "":
                continue
            count_dict[a1, a2] += 1.0*score
            count_dict[a2, a1] += 1.0*score
            sound_dict[a1] += 2.0*score
            sound_dict[a2] += 2.0*score

    log_weight = 2 * np.log(sum(list(
        sound_dict.values()))) - np.log(sum(list(
            count_dict.values())))

    for (c1, c2) in count_dict.keys():
        m = count_dict[c1, c2]
        #assert m > 0

        num = np.log(m)
        denom = np.log(sound_dict[c1]) + np.log(sound_dict[c2])
        val = num - denom + log_weight
        count_dict[c1, c2] = val

    return count_dict


class OnlinePMITrainer:
    """Train a PMI scorer step-by-step on always improving alignments."""

    def __init__(self, margin=1.0, alpha=0.75, gop=-2.5, gep=-1.75):
        """Create a persistent aligner object.

        margin: scaling factor for scores
        alpha: Decay in update weight (must be between 0.5 and 1)
        gop, gep: Gap opening and extending penalty. gop=None uses character-dependent penalties.

        """
        self.margin = margin
        self.alpha = alpha
        self.n_updates = 0
        self.pmidict = collections.defaultdict(float)
        self.gep = gep
        self.gop = gop

    def align_pairs(self, word_pairs, local=False):
        """Align a list of word pairs, removing those that align badly."""
        algn_list, scores = [], []
        n_zero = 0
        for w in range(len(word_pairs)-1, -1, -1):
            ((c1, l1, w1), (c2, l2, w2)) = word_pairs[w]
            s, alg = distances.needleman_wunsch(
                w1, w2, gop=self.gop, gep=self.gep, lodict=self.pmidict,
                local=local)
            if s <= self.margin:
                n_zero += 1
                word_pairs.pop(w)
                continue
            algn_list.append(alg)
            scores.append(s)
        self.update_pmi_dict(algn_list, scores=scores)
        return algn_list, n_zero

    def update_pmi_dict(self, algn_list, scores=None):
        eta = (self.n_updates + 2) ** (-self.alpha)
        for k, v in calc_pmi(algn_list, scores).items():
            pmidict_val = self.pmidict.get(k, 0.0)
            self.pmidict[k] = (eta * v) + ((1.0 - eta) * pmidict_val)
        self.n_updates += 1


class MaxPairDict(dict):
    """A maximum-of-pairs lookup dictionary.

    Multiple options must be given in a tuple, for historical
    reasons. (A set would be much nice, because it only contains
    hashables but isn't itself hashable, so there would be no danger
    of confusion.

    >>> m = MaxPairDict({(1, 1): 2, (1, 0): 1, (0, 0): 0})
    >>> m[(1, 1)]
    2
    >>> m[((0, 1), 0)]
    1
    >>> m[((0, 1), (0, 1))]
    Traceback (most recent call last):
    ...
    KeyError: (0, 1)

    """

    def __getitem__(self, key):
        """Return the maximum value among all pairs given.

        x.__getitem__(y) <=> x[y]
        """
        key1, key2 = key
        max_val = -float('inf')
        if type(key1) != tuple:
            key1 = [key1]
        if type(key2) != tuple:
            key2 = [key2]
        for k1 in key1:
            for k2 in key2:
                v = dict.__getitem__(self, (k1, k2))
                if v > max_val:
                    max_val = v
            return max_val

    def get(self, key, default=None):
        """Return m[k] if k in m, otherwise default (None).

        Ideally, at some point, this will work:
        >>> m = MaxPairDict({(1, 1): 2, (1, 0): 1, (0, 0): 0})
        >>> m.get(((0, 1), (0, 1)))
        2

        Currently, this returns None, because one of the values cannot
        be found.

        """
        try:
            return self[key]
        except KeyError:
            return default


def multi_align(
        similarity_sets, guide_tree, pairwise=distances.needleman_wunsch,
        **kwargs):
    """Align multiple sequences according to a give guide tree."""
    languages = {leaf.name: leaf for leaf in guide_tree.get_leaves()}
    for s, similarityset in enumerate(similarity_sets):
        for (language, concept, form) in similarityset:
            try:
                leaf = languages[language]
            except KeyError:
                continue
            try:
                leaf.forms
            except AttributeError:
                leaf.forms = {}
            leaf.forms.setdefault(s, []).append(
                    ((language, ), (concept, ), tuple((x, ) for x in form)))

    for node in guide_tree.walk('postorder'):
        print(node.name)
        try:
            entries_by_group = node.forms
        except AttributeError:
            entries_by_group = {}
            for child in node.descendants:
                try:
                    for group, alignment in child.alignment.items():
                        entries_by_group.setdefault(group, []).append(alignment)
                except AttributeError:
                    pass
        aligned_groups = {}
        for group in entries_by_group:
            forms = entries_by_group[group]

            already_aligned = None
            for (new_languages, new_concepts, new_alignment) in forms:
                if not already_aligned:
                    languages = new_languages
                    concepts = new_concepts
                    already_aligned = new_alignment
                else:
                    gap1 = ('', ) * len(languages)
                    gap2 = ('', ) * len(new_alignment[0])

                    languages += new_languages
                    concepts += new_concepts

                    print("Aligning:")
                    print(already_aligned)
                    print(new_alignment)
                    s, combined_alignment = pairwise(
                        already_aligned, new_alignment, **kwargs)
                    print(combined_alignment)
                    already_aligned = tuple(
                        (x if x else gap1) + (y if y else gap2)
                        for x, y in combined_alignment)
                    print(already_aligned)
            aligned_groups[group] = languages, concepts, already_aligned
        node.alignment = aligned_groups
    return node.alignment



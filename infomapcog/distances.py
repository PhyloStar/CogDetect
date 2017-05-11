import collections
import itertools
import numpy as np

def sigmoid(score):
    return 1.0/(1.0+np.exp(score))

def dice(a, b):
    la = len(a) - 1;lb = len(b) - 1
    overlap = 0
    dicta = defaultdict(int)
    dictb = defaultdict(int)
    for i in range(len(a) - 1):
        tmp = ",".join(map(str, a[i:i + 2]))
        dicta[tmp] += 1
    for j in range(len(b) - 1):
        tmp = ",".join(map(str, b[j:j + 2]))
        dictb[tmp] += 1
    for entry in dicta:
        if(dictb.has_key(entry)):
            overlap = overlap + min(dicta.get(entry), dictb.get(entry))
    total = la + lb
    if total == 0:
        return 0
    if UNNORM:
        return float(2.0*overlap)
    return float(total) - float(2.0*overlap)

def normalized_leventsthein(a, b):
    """
    Leventsthein distance normalized
    :param a: word
    :type a: str
    :param b: word
    :type b: str
    :return: distance score
    :rtype: float
    """
    m = [];
    la = len(a) + 1;
    lb = len(b) + 1
    for i in range(0, la):
        m.append([])
        for j in range(0, lb): m[i].append(0)
        m[i][0] = i
    for i in range(0, lb): m[0][i] = i
    for i in range(1, la):
        for j in range(1, lb):
            s = m[i - 1][j - 1]
            if (a[i - 1] != b[j - 1]): s = s + 1
            m[i][j] = min(m[i][j - 1] + 1, m[i - 1][j] + 1, s)
    la = la - 1;
    lb = lb - 1
    return float(m[la][lb])/ float(max(la, lb))


def LD(x, y, lodict={}):
    """Standard NW alignment.

    Needleman-Wunsch algorithm for pairwise string alignment with
    default gap penalties.

    """
    return needleman_wunsch(x, y, lodict=lodict, gop=-1, gep=-1)


def needleman_wunsch(x, y, lodict={}, gop=-2.5, gep=-1.75, local=False, indel=''):
    """Needleman-Wunsch algorithm with affine gaps penalties.

    This code implements the NW algorithm for pairwise string
    alignment with affine gap penalties.

    'lodict' must be a dictionary with all symbol pairs as keys and
    match scores as values, or a False value (including an empty
    dictionary) to denote (-1, 1) scores. gop and gep are gap
    penalties for opening/extending a gap; alternatively, you can set
    'gop' to None and provide element/gap alignment costs.
    indel takes the character used to denote an indel.

    Returns the alignment score and one optimal alignment.

    >>> needleman_wunsch("AAAAABBBB", "AACAABBCB")
    (5.0, [('A', 'A'), ('A', 'A'), ('A', 'C'), ('A', 'A'), ('A', 'A'), ('B', 'B'), ('B', 'B'), ('B', 'C'), ('B', 'B')])
    >>> needleman_wunsch("banana", "mancala", local=True)
    (2.0, [('a', 'a'), ('n', 'n')])
    >>> needleman_wunsch("abc", "t", lodict={('a', ''): 0, ('b', ''): -2, ('c', ''): -0.5}, gop=None)
    (-1.5, [('a', ''), ('b', 't'), ('c', '')])

    """
    n, m = len(x), len(y)
    dp = np.zeros((n + 1, m + 1))
    pointers = np.zeros((n + 1, m + 1), np.int32)
    if not local:
        for i1, c1 in enumerate(x):
            if gop is None:
                dp[i1 + 1, 0] = lodict.get((c1, indel), gep)
            else:
                dp[i1 + 1, 0] = dp[i1, 0]+(gep if i1 + 1 > 1 else gop)
            pointers[i1 + 1, 0] = 1
        for i2, c2 in enumerate(y):
            if gop is None:
                dp[0, i2 + 1] = lodict.get((indel, c2), gep)
            else:
                dp[0, i2 + 1] = dp[0, i2]+(gep if i2 + 1 > 1 else gop)
            pointers[0, i2 + 1] = 2
    for i1, c1 in enumerate(x):
        for i2, c2 in enumerate(y):
            match = dp[i1, i2] + lodict.get(
                (c1, c2),
                1 if c1 == c2 else -1)
            insert = dp[i1, i2 + 1] + (
                lodict.get((c1, indel), gep) if gop is None else
                gep if pointers[i1, i2 + 1] == 1 else gop)
            delet = dp[i1 + 1, i2] + (
                lodict.get((indel, c2), gep) if gop is None else
                gep if pointers[i1 + 1, i2] == 2 else gop)
            pointers[i1 + 1, i2 + 1] = p = np.argmax([match, insert, delet])
            max_score = [match, insert, delet][p]
            if local and max_score < 0:
                max_score = 0
            dp[i1 + 1, i2 + 1] = max_score
    alg = []
    if local:
        i, j = np.unravel_index(dp.argmax(), dp.shape)
    else:
        i, j = n, m
    score = dp[i, j]
    while (i > 0 or j > 0):
        pt = pointers[i, j]
        if pt == 0:
            i -= 1
            j -= 1
            alg = [(x[i], y[j])] + alg
        if pt == 1:
            i -= 1
            alg = [(x[i], indel)] + alg
        if pt == 2:
            j -= 1
            alg = [(indel, y[j])] + alg
        if local and dp[i, j] == 0:
            break
    return score, alg

def prefix(a, b, normalized=True):
    la = len(a); lb = len(b)
    minl = min(la,lb)
    maxl = max(la,lb)
    pref = 0
    for i in range(minl):
        if a[i] == b[i]:
            pref += 1
    if not normalized:
        return float(pref)
    return float(maxl) - float(pref)

def ident(a,b):
    overlap = 0
    if a == b :
        overlap = 1
    else:
        overlap = 0
    return 1.0 - float(overlap)


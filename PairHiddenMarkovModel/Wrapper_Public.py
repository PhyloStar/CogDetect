# -*- coding: utf-8 -*-
import collections
import itertools

import numpy as np

import PairHiddenMarkovModel.PHMM_Public as PHMM_Public


def read_data(wordpair_file, exc=("%", "~", "*", "$", "\"")):
    """
    This function reads a tab separated data file. The file should have three columns and the first line should be the
    header. The data should come in the following format:

    language	gloss   transcription
    eng hand    hEnt
    ...

    :param wordpair_file: filename of the data file
    :type wordpair_file: str
    :param exc: characters that should be excluded
    :type exc: list or tuple
    :return: file content as list and alphabet as list
    :rtype: (list of list, dict of (str, int) )
    """

    with open(wordpair_file, "r") as infile:
        cont = infile.readlines()
    # remove header
    cont.pop(0)
    data = []
    while cont:
        data.append(cont.pop(0).split("\t"))

    # exclude special characters
    data = [[i[0], i[1], tuple([j for j in i[2] if j not in exc])] for i in data]

    # remove empty data points
    data = [i for i in data if len(i[2]) != 0]

    # create alphabet
    alpha_list = list(set([j for i in data for j in i[2]]))
    alphabet = collections.defaultdict()

    for ind, symb in enumerate(alpha_list):
        alphabet[symb] = ind
    return data, alphabet


def concept_clusters(data):
    """
    This functions clusters the data according to the concepts
    :param data: list of lists with iso, concept, asjp representation
    :type data: list
    :return: dictionary sorted by concept
    :rtype: dict
    """

    clusters = collections.defaultdict(list)
    for i in data:
        clusters[i[1]].append([i[0], i[2]])

    return clusters


def create_word_pairs(clusters):
    """
    Create word pairs from concept clusters
    :param clusters: dictionary of words clustered by same concept
    :type clusters: dict
    :return: list of word pairs
    :rtype: list of tuple
    """
    wpairs = []
    for key, value in list(clusters.items()):
        for p1, p2 in itertools.combinations(value, 2):
            wpairs.append((p1[1], p2[1]))
    return wpairs


def chunks(some_list, n):
    """
    Yield successive size-sized chunks from some_list.
    :param some_list: list to be chunked
    :type some_list: list
    :param n: size of the chunks
    :type n: int
    :return: Generator returning chunks of size n
    :rtype: list
    """

    for i in range(0, len(some_list), n):
        yield some_list[i:i + n]


def merge(mat1, mat2, run, a):
    """
    Merge two matrices or vectors
    :param mat1: matrix or vector
    :type mat1: numpy.core.ndarray
    :param mat2: matrix or vector
    :type mat2: numpy.core.ndarray
    :param run: number of run
    :type run: int or float
    :param a: scaling parameter
    :type a: float
    :return: matrix or vector
    :rtype: numpy.core.ndarray
    """

    eta = np.power(run + 2, -a)

    return np.multiply(1 - eta, mat1) + np.multiply(eta, mat2)


def training_wrapped(data_file):
    """
    This function wrapps Batch EM
    :param data_file: file containing training data
    :type data_file: str
    :return: trained parameters, emission matrix, gap x, gap y, Transition
    :rtype: (np.core.ndarray, np.core.ndarray, np.core.ndarray, np.core.ndarray)
    """
    data, alphabet = read_data(wordpair_file=data_file)
    wordpairs = create_word_pairs(clusters=concept_clusters(data=data))

    # create storage for new parameters, include some pseudo counts to facilitate normalization
    em_store = np.zeros(len(list(alphabet.keys())), len(list(alphabet.keys())))
    em_store[:] = 0.0001

    g_store = np.zeros(len(list(alphabet.keys())))
    g_store[:] = 0.0001

    trans_store = np.array([10.0001, 10.0001, 0.0, 10.0001, 10.0001, 10.0001, 10.0001])

    # create initial parameters
    em_input = np.ones(len(list(alphabet.keys())), len(list(alphabet.keys())))
    em_input /= np.sum(em_input)

    gx_input = np.ones(len(list(alphabet.keys())))
    gx_input /= np.sum(gx_input)

    gy_input = np.ones(len(list(alphabet.keys())))
    gy_input /= np.sum(gy_input)

    # delta, epsilon, lambda, taum, tauxy
    trans_input = np.array([0.3, 0.3, 0.0, 0.1, 0.1])
    converged = False
    while converged is False:
        new_em, new_gx, new_gy, new_trans = PHMM_Public.baum_welch_train(list_of_seq=wordpairs,
                                                                         em_probs=em_input,
                                                                         gap_probs_x=gx_input,
                                                                         gap_probs_y=gy_input,
                                                                         trans_probs=trans_input,
                                                                         new_em=em_store,
                                                                         new_g_probs=g_store,
                                                                         new_trans=trans_store,
                                                                         alphabet=alphabet)

        results = [np.allclose(em_input, new_em), np.allclose(gx_input, new_gx),
                   np.allclose(gy_input, new_gy), np.allclose(trans_input, new_trans)]

        # new model parameters
        em_input = new_em
        gx_input = new_gx
        gy_input = new_gy
        trans_input = new_trans

        if False not in results:
            converged = True

    return em_input, gy_input, gy_input, trans_input


def training_wrapped_online(data_file, size, alpha):
    """
    This function wraps the online EM training
    :param data_file: file containing training data
    :type data_file: str
    :param size: chunk size for online EM
    :type size: int
    :param alpha: update strength parameters
    :type alpha: float
    :return: trained parameters, emission matrix, gap x, gap y, Transition
    :rtype: (np.core.ndarray, np.core.ndarray, np.core.ndarray, np.core.ndarray)
    """

    data, alphabet = read_data(wordpair_file=data_file)
    wordpairs = create_word_pairs(clusters=concept_clusters(data=data))

    # create storage for new parameters, include some pseudo counts to facilitate normalization
    em_store = np.zeros(len(list(alphabet.keys())), len(list(alphabet.keys())))
    em_store[:] = 0.0001

    g_store = np.zeros(len(list(alphabet.keys())))
    g_store[:] = 0.0001

    trans_store = np.array([10.0001, 10.0001, 0.0, 10.0001, 10.0001, 10.0001, 10.0001])

    # create initial parameters
    em_input = np.ones(len(list(alphabet.keys())), len(list(alphabet.keys())))
    em_input /= np.sum(em_input)

    gx_input = np.ones(len(list(alphabet.keys())))
    gx_input /= np.sum(gx_input)

    gy_input = np.ones(len(list(alphabet.keys())))
    gy_input /= np.sum(gy_input)

    # delta, epsilon, lambda, taum, tauxy
    trans_input = np.array([0.3, 0.3, 0.0, 0.1, 0.1])

    n_o_batches = 0.0
    converged = False

    while converged is False:

        np.random.shuffle(wordpairs)
        word_pairs = chunks(wordpairs, size)

        em_check = em_input
        gx_check = gx_input
        gy_check = gy_input
        trans_check = trans_input

        for chunk in word_pairs:
            new_em, new_gx, new_gy, new_trans = PHMM_Public.baum_welch_train(list_of_seq=chunk,
                                                                             em_probs=em_input,
                                                                             gap_probs_x=gx_input,
                                                                             gap_probs_y=gy_input,
                                                                             trans_probs=trans_input,
                                                                             new_em=em_store,
                                                                             new_g_probs=g_store,
                                                                             new_trans=trans_store,
                                                                             alphabet=alphabet)

            em_input = merge(em_input, new_em, n_o_batches, alpha)
            gx_input = merge(gx_input, new_gx, n_o_batches, alpha)
            gy_input = merge(gy_input, new_gy, n_o_batches, alpha)
            trans_input = merge(trans_input, new_trans, n_o_batches, alpha)

        results = [np.allclose(em_check, em_input), np.allclose(gx_check, gx_input),
                   np.allclose(gy_check, gy_input), np.allclose(trans_check, trans_input)]

        if False not in results:
            converged = True

    return em_input, gx_input, gy_input, trans_input


def alignment_wrapped(data_file, em, gx, gy, trans, equilibrium):
    """
    This function returns the alignment scores of the words in the data file
    :param data_file: file containing words to be aligned
    :type data_file: str
    :param em: emission probabilities
    :type em: np.core.ndarray
    :param gx: gap probabilities for state x
    :type gx: np.core.ndarray
    :param gy: gap probabilities for state y
    :type gy: np.core.ndarray
    :param trans: transition probabilities
    :type trans: np.core.ndarray
    :param equilibrium: equilibrium probabilities for random model
    :type equilibrium: np.core.ndarray
    :return: dictionary of alignment scores
    :rtype: dict
    """

    data, alphabet = read_data(wordpair_file=data_file)
    wordpairs = create_word_pairs(clusters=concept_clusters(data=data))

    score_dict = collections.defaultdict()
    for w1, w2 in wordpairs:
        v_score = PHMM_Public.viterbi(seq1=w1,
                                      seq2=w2,
                                      em_probs=em,
                                      gap_probs_x=gx,
                                      gap_probs_y=gy,
                                      trans_probs=trans)[1]
        r_score = PHMM_Public.random_model(seq1=w1,
                                           seq2=w2,
                                           eq_probs=equilibrium)
        score_dict[(w1, w2)] = v_score / r_score

    return score_dict

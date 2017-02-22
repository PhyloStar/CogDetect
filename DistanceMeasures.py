
import collections
import itertools
import numpy as np

def ldn(a, b):
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
    return float(m[la][lb]) / float(max(la, lb))






def upgma(distmat, threshold, names):
    """
    UPGMA
    :param distmat: distance matrix
    :type distmat: list or numpy.core.ndarray
    :param threshold: threshold for cutting the treee
    :type threshold: float
    :param names: name of the taxa
    :type names: list
    :return: clusters
    :rtype: dict
    """

    # create cluster for individual nodes
    clusters = collections.defaultdict(list)
    for i in range(len(distmat)):
        clusters[i] = [i]

    # call internal upgma
    clust = upgma_int(clusters, distmat, threshold)
    
    # assign names to the clusters
    for key in clust:
        clust[key] = [names[i] for i in clust[key]]
    return clust


def upgma_int(
        clusters,
        matrix,
        threshold
        ):
    """
    Internal upgma implementation
    :param clusters: dictionary of clusters
    :type clusters: dict
    :param matrix: distance matrix
    :type matrix: list or numpy.core.ndarry
    :param threshold: threshold for cutting the upgma tree
    :type threshold: float
    :return: clusters
    :rtype: dict
    """

    done = False

    while done is False:

        # check if first termination condition is reached
        if len(clusters) == 1:
            done = True

        else:
            # dictionary with indices of scores
            sc_ind = collections.defaultdict(float)
            # calculate score of existing clusters
            for (i, valA), (j, valB) in itertools.permutations(clusters.items(), 2):
                s = 0.0
                ct = 0
                for vA, vB in itertools.product(valA, valB):
                    s += matrix[vA][vB]
                    ct += 1
                sc_ind[(i, j)] = (s / ct)

            minimum_ind = min(sc_ind, key=sc_ind.get)

            # check if second termination condition is reached
            # everything left above threshold
            if sc_ind[minimum_ind] <= threshold:
                # form new cluster
                idx, jdx = minimum_ind
                clusters[idx] += clusters[jdx]
                del clusters[jdx]
            else:
                done = True

    return clusters


def single_linkage(distmat, threshold, names):
    """
    single linkage clustering
    :param distmat: distance matrix
    :type distmat: list or numpy.core.ndarray
    :param threshold: threshold for cutting the treee
    :type threshold: float
    :param names: name of the taxa
    :type names: list
    :return: clusters
    :rtype: dict
    """

    # create cluster for individual nodes
    clusters = collections.defaultdict(list)
    for i in range(len(distmat)):
        clusters[i] = [i]

    # call internal upgma
    clust = single_linkage_int(clusters, distmat, threshold)

    # assign names to the clusters
    for key in clust:
        clust[key] = [names[i] for i in clust[key]]
    return clust


def single_linkage_int(clusters, matrix, threshold):
    """
    internal implementation of single linkage clustering
    :param clusters: dictionary of clusters
    :type clusters: dict
    :param matrix: distance matrix
    :type matrix: list or numpy.core.ndarry
    :param threshold: threshold for cutting the upgma tree
    :type threshold: float
    :return: clusters
    :rtype: dict
    """
    done = False

    while done is False:

        # check if first termination condition is reached
        if len(clusters) == 1:
            done = True

        else:
            # dictionary with indices of scores
            sc_ind = collections.defaultdict(float)
            # calculate score of existing clusters
            for (i, valA), (j, valB) in itertools.permutations(clusters.items(), 2):
                sc_ind[(i, j)] = float("inf")
                for vA, vB in itertools.product(valA, valB):
                    if matrix[vA][vB] < sc_ind[(i, j)]:
                        sc_ind[(i, j)] = matrix[vA][vB]

            minimum_ind = min(sc_ind, key=sc_ind.get)

            # check if second termination condition is reached
            # everything left above threshold
            if sc_ind[minimum_ind] <= threshold:
                # form new cluster
                idx, jdx = minimum_ind
                clusters[idx] += clusters[jdx]
                del clusters[jdx]
            else:
                done = True

    return clusters



def b_cubed(true_labels, labels):
    d = collections.defaultdict()
    precision = [0.0]*len(true_labels)
    recall = [0.0]*len(true_labels)

    for t, l in zip(true_labels, labels):
        d[str(l)] = t

    for i, l in enumerate(labels):
        match = 0.0
        prec_denom = 0.0
        recall_denom = 0.0
        for j, m in enumerate(labels):
            if l == m:
                prec_denom += 1.0
                if true_labels[i] == true_labels[j]:
                    match += 1.0
                    recall_denom += 1.0
            elif l != m:
                if true_labels[i] == true_labels[j]:
                    recall_denom += 1.0
        precision[i] = match/prec_denom
        recall[i] = match/recall_denom
    #print precision, recall
    avg_precision = np.average(precision)
    avg_recall = np.average(recall)
    avg_f_score = 2.0*avg_precision*avg_recall/(avg_precision+avg_recall)
    return avg_precision, avg_recall,avg_f_score

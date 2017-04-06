
import collections
import itertools as it
import numpy as np
import igraph
from collections import defaultdict
import distances



def igraph_clustering(matrix, threshold, method='labelprop'):
    """
    Method computes Infomap clusters from pairwise distance data.
    """

    G = igraph.Graph()
    vertex_weights = []
    for i in range(len(matrix)):
        G.add_vertex(i)
        vertex_weights += [0]
    
    # variable stores edge weights, if they are not there, the network is
    # already separated by the threshold
    weights = None
    for i,row in enumerate(matrix):
        for j,cell in enumerate(row):
            if i < j:
                if cell <= threshold:
                    G.add_edge(i, j, weight=1-cell, distance=cell)
                    weights = 'weight'

    if method == 'infomap':
        comps = G.community_infomap(edge_weights=weights,
                vertex_weights=None)
        
    elif method == 'labelprop':
        comps = G.community_label_propagation(weights=weights,
                initial=None, fixed=None)

    elif method == 'ebet':
        dg = G.community_edge_betweenness(weights=weights)
        oc = dg.optimal_count
        comps = False
        while oc <= len(G.vs):
            try:
                comps = dg.as_clustering(dg.optimal_count)
                break
            except:
                oc += 1
        if not comps:
            print('Failed...')
            comps = list(range(len(G.sv)))
            input()
    elif method == 'multilevel':
        comps = G.community_multilevel(return_levels=False)
    elif method == 'spinglass':
        comps = G.community_spinglass()

    D = {}
    for i,comp in enumerate(comps.subgraphs()):
        vertices = [v['name'] for v in comp.vs]
        for vertex in vertices:
            D[vertex] = i+1

    return D
    
def infomap_concept_evaluate_scores(d, lodict, gop, gep, threshold, cogid_dict):
    #fout = open("output.txt","w")
    average_fscore = []
    f_scores = []#defaultdict(list)
    n_clusters = 0
    for concept in d:
        ldn_dist_dict = defaultdict(lambda: defaultdict(float))
        langs = list(d[concept].keys())
        if len(langs) == 1:
            print(concept)
            continue
        scores, cognates = [], []
        #ex_langs = list(set(lang_list) - set(langs))
        for l1, l2 in it.combinations(langs, r=2):
            if d[concept][l1].startswith("-") or d[concept][l2].startswith("-"): continue
            w1, w2 = d[concept][l1], d[concept][l2]
            score = distances.needleman_wunsch(
                w1, w2, lodict=lodict, gop=gop, gep=gep)[0]
            score = 1.0 - (1.0/(1.0+np.exp(-score)))
            ldn_dist_dict[l1][l2] = score
            ldn_dist_dict[l2][l1] = ldn_dist_dict[l1][l2]
        distMat = np.array([[ldn_dist_dict[ka][kb] for kb in langs] for ka in langs])
        clust = igraph_clustering(distMat, threshold, method='labelprop')
        
        
        predicted_labels = defaultdict()
        predicted_labels_words = defaultdict()
        for k, v in clust.items():
            predicted_labels[langs[k]] = v
            predicted_labels_words[langs[k],d[concept][langs[k]]] = v
        
        print(concept,"\n",predicted_labels_words)
        predl, truel = [], []
        for l in langs:
            truel.append(cogid_dict[concept][l])
            predl.append(predicted_labels[l])
        scores = b_cubed(truel, predl)
        
        #scores = metrics.f1_score(truel, predl, average="micro")
        print(concept, len(langs), scores, len(set(clust.values())), len(set(truel)), "\n")
        f_scores.append(list(scores))
        n_clusters += len(set(clust.values()))
        #t = utils.dict2binarynexus(predicted_labels, ex_langs, lang_list)
        #print(concept, "\n",t)
        #print("No. of clusters ", n_clusters)
    #print(np.mean(np.array(f_scores), axis=0))
    f_scores = np.mean(np.array(f_scores), axis=0)
    print(f_scores[0], f_scores[1], 2.0*f_scores[0]*f_scores[1]/(f_scores[0]+f_scores[1]))


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
            for (i, valA), (j, valB) in it.permutations(clusters.items(), 2):
                s = 0.0
                ct = 0
                for vA, vB in it.product(valA, valB):
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
            for (i, valA), (j, valB) in it.permutations(clusters.items(), 2):
                sc_ind[(i, j)] = float("inf")
                for vA, vB in it.product(valA, valB):
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

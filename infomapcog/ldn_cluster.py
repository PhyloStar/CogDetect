from collections import defaultdict
import itertools as it
import sys, distances, igraph, utils
import numpy as np
import random, codecs
import DistanceMeasures as DM
from sklearn import metrics
random.seed(1234)

MAX_ITER = 10
tolerance = 0.001
infomap_threshold = 0.5
min_batch = 128
margin = 0.0

dataname = sys.argv[1]
#fname = sys.argv[2]
char_list = []

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
    return w

def read_data_ielex_type(fname):
    line_id = 0
    data_dict = defaultdict(lambda : defaultdict())
    cogid_dict = defaultdict(lambda : defaultdict())
    words_dict = defaultdict(lambda : defaultdict(list))
    langs_list = []
    #f = codecs.open(fname, "r", "utf8")
    f = open(fname)
    f.readline()
    for line in f:
        line = line.strip()
        arr = line.split("\t")
        lang = arr[0]
        if "Proto" in lang: continue
        #if lang in ["ELFDALIAN", "GUTNISH_LAU", "STAVANGERSK"]:
        #    continue
        concept = arr[2]
        cogid = arr[6]
        cogid = cogid.replace("-","")
        cogid = cogid.replace("?","")
        asjp_word = clean_word(arr[5].split(",")[0])
        if "/" in asjp_word: continue

        for ch in asjp_word:
            if ch not in char_list:
                char_list.append(ch)

        if len(asjp_word) < 1:
            continue
        if "K" in asjp_word:
            print(line)
        data_dict[concept][line_id,lang] = asjp_word
        cogid_dict[concept][line_id,lang] = cogid
        words_dict[concept][lang].append(asjp_word)
        if lang not in langs_list:
            langs_list.append(lang)
        line_id += 1
    f.close()
    print(list(data_dict.keys()))
    return (data_dict, cogid_dict, words_dict, langs_list)

def igraph_clustering(matrix, threshold, method='infomap'):
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
    
def infomap_concept_evaluate_scores(d, lodict, gop, gep):
    #fout = open("output.txt","w")
    average_fscore = []
    f_scores = []#defaultdict(list)
    bin_mat, n_clusters = None, 0
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
            score = distances.ldn(w1, w2)
            ldn_dist_dict[l1][l2] = score
            ldn_dist_dict[l2][l1] = ldn_dist_dict[l1][l2]
        distMat = np.array([[ldn_dist_dict[ka][kb] for kb in langs] for ka in langs])
        clust = igraph_clustering(distMat, infomap_threshold, method='labelprop')
        
        
        predicted_labels = defaultdict()
        predicted_labels_words = defaultdict()
        for k, v in clust.items():
            predicted_labels[langs[k]] = v
            predicted_labels_words[langs[k],d[concept][langs[k]]] = v
        
        #print(concept,"\n",predicted_labels_words)
        predl, truel = [], []
        for l in langs:
            truel.append(cogid_dict[concept][l])
            predl.append(predicted_labels[l])
        scores = DM.b_cubed(truel, predl)
        
        #scores = metrics.f1_score(truel, predl, average="micro")
        print(concept, len(langs), scores, len(set(clust.values())), len(set(truel)), "\n")
        f_scores.append(list(scores))
        #n_clusters += len(set(clust.values()))
        #t = utils.dict2binarynexus(predicted_labels, ex_langs, lang_list)

    #print(np.mean(np.array(f_scores), axis=0))
    f_scores = np.mean(np.array(f_scores), axis=0)
    print(f_scores[0], f_scores[1], 2.0*f_scores[0]*f_scores[1]/(f_scores[0]+f_scores[1]))
    return bin_mat


data_dict, cogid_dict, words_dict, langs_list = read_data_ielex_type(dataname)
print("Character list \n\n", char_list)


bin_mat = infomap_concept_evaluate_scores(data_dict, None, -2.5, -1.75)
#nchar, nlangs = np.array(bin_mat).shape

sys.exit(1)
print("begin data;")
print("   dimensions ntax=", str(nlangs), "nchar=", str(nchar), ";\nformat datatype=restriction interleave=no missing= ? gap=-;\nmatrix\n")

for row, lang in zip(np.array(bin_mat).T, lang_list):
    #print(row,len(row), "\n")
    rowx = "".join([str(x) for x in row])
    print(lang, "\t", rowx.replace("2","?"))
print(";\nend;")

    
    
    
    

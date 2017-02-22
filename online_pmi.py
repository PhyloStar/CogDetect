from collections import defaultdict
import itertools as it
import sys, distances, igraph, utils
import numpy as np
import random, codecs
import DistanceMeasures as DM
from sklearn import metrics
from lingpy import *
random.seed(1234)

##TODO: Add a ML based estimation of distance or a JC model for distance between two sequences
##Separate clustering code.
##Add doculect distance as regularization

MAX_ITER = 15
tolerance = 0.001
infomap_threshold = 0.5
min_batch = 256
margin = 1.0

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
    w = w.replace("ᶢ","")
    w = w.replace("C","c")
    w = w.replace("L","l")
    w = w.replace("W","w")
    w = w.replace("T","t")
    return w

def ipa2sca(w):
    return "".join(tokens2class(ipa2tokens(w), 'asjp')).replace("0","")

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
        #if lang in ["ELFDALIAN", "GUTNISH_LAU", "STAVANGERSK"]:
        #    continue
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
    f.close()
    print(list(data_dict.keys()))
    return (data_dict, cogid_dict, words_dict, langs_list)

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
            score = distances.nw(w1, w2, lodict=lodict, gp1=gop, gp2=gep)[0]
            score = 1.0 - (1.0/(1.0+np.exp(-score)))
            ldn_dist_dict[l1][l2] = score
            ldn_dist_dict[l2][l1] = ldn_dist_dict[l1][l2]
        distMat = np.array([[ldn_dist_dict[ka][kb] for kb in langs] for ka in langs])
        clust = igraph_clustering(distMat, infomap_threshold, method='labelprop')
        
        
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
        scores = DM.b_cubed(truel, predl)
        
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
    return bin_mat

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

data_dict, cogid_dict, words_dict, langs_list = read_data_ielex_type(dataname)
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
bin_mat = infomap_concept_evaluate_scores(data_dict, pmidict, -2.5, -1.75)
#nchar, nlangs = np.array(bin_mat).shape

sys.exit(1)
print("begin data;")
print("   dimensions ntax=", str(nlangs), "nchar=", str(nchar), ";\nformat datatype=restriction interleave=no missing= ? gap=-;\nmatrix\n")

for row, lang in zip(np.array(bin_mat).T, lang_list):
    #print(row,len(row), "\n")
    rowx = "".join([str(x) for x in row])
    print(lang, "\t", rowx.replace("2","?"))
print(";\nend;")

    
    
    
    

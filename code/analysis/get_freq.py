__author__ = 'mauceri2'

__author__ = 'mauceri2'
import numpy as np
import scipy.sparse as sparse
import scipy.io as spio
import itertools
import os

def get_freq(comatrix_path, countstr, knowledge_root, save_description, candidate_vocab, image_threshold):
    co_object = spio.loadmat(comatrix_path)

    #Mask diagonal
    comatrix = co_object['comatrix_{}s'.format(countstr)]
    comatrix[np.eye(np.shape(comatrix)[0], dtype=bool)] = 0


    concepts = co_object['concept_vocabulary']
    concept_ind = {concepts[ii].strip(): ii for ii in range(0, np.shape(concepts)[0])}
    candidate_delim = [c.split() for c in candidate_vocab]
    candidate_ind = [[concept_ind[c] for c in delim] for delim in candidate_delim]
    candidate_ind = [[sorted([combo[0], combo[1]]) for combo in itertools.combinations(c, 2)] for c in candidate_ind]
    candidate_frequency = [min([comatrix[combo[0], combo[1]] for combo in c]) for c in candidate_ind]

    candidate_frequency = sorted(zip(candidate_vocab, candidate_frequency), key=lambda x: x[1])
    top_candidate_frequency = [t for t in candidate_frequency if t[1] > image_threshold]

    new_vocabulary_frequency = [t for t in top_candidate_frequency if len(t[0].split()) == 2]
    filtered_candidates = [t for t in top_candidate_frequency if len(t[0].split()) > 2]

    excluded_vocab = [t for t in candidate_frequency if t[1] <= image_threshold]
    excluded_vocab_formated = ["{}\t{}".format(t[0], t[1]) for t in excluded_vocab]
    save_path = os.path.join(knowledge_root, "Freebase", "candidate_concepts", save_description + "_no_support.txt")
    with open(save_path, 'a') as f:
        f.write("\n".join(excluded_vocab_formated))
        f.write("\n")

    return new_vocabulary_frequency, filtered_candidates



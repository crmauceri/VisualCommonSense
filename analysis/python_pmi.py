__author__ = 'mauceri2'
import numpy as np
import scipy.sparse as sparse
import scipy.io as spio
import os

def pmi(A, totalPerRow, total):
    (w, h) = np.shape(totalPerRow)
    pxy = A/total #joint probability
    px = totalPerRow/total #probability of x
    px_mat = np.tile(px, (1, w))
    py_mat = np.tile(px.transpose(), (w, 1))

    denominator = np.multiply(px_mat, py_mat)

    return np.divide(np.log2(np.divide(pxy, denominator)), -1 * np.log2(pxy))


def high_pmi(comatrix_path, pmi_threshold, image_threshold, countstr, additional_mask, vocabulary_list):
    co_object = spio.loadmat(comatrix_path)

    comatrix = (co_object['comatrix_{}s'.format(countstr)]).astype(np.float64)
    total_per_concept = (co_object['vocab_{}_count'.format(countstr)]).astype(np.float64)
    total = (co_object['total_{}s'.format(countstr)]).astype(np.float64)

    if sparse.issparse(comatrix):
        comatrix = comatrix.todense()
    if sparse.issparse(total_per_concept):
        total_per_concept = total_per_concept.todense()

    #Mask diagonal
    comatrix[np.eye(np.shape(comatrix)[0], dtype=bool)] = 0

    #Use Laplace/Addative smoothing
    #Pretend we have one image with all the tags
    if len(np.where(comatrix == 0))>0:
        num_vocab = np.shape(comatrix)[0]
        comatrix = comatrix + 1
        total_per_concept = total_per_concept + 1
        total = total + 1

    co_pmi = pmi(comatrix, total_per_concept, total)

    if np.max(co_pmi) <= 1.0:
        mask = np.greater(co_pmi, pmi_threshold) & np.greater(comatrix, image_threshold) & np.greater(additional_mask, 0)

        concept_ind_x, concept_ind_y = np.nonzero(mask)
        concept_ind_x = np.reshape(concept_ind_x, -1)
        concept_ind_y = np.reshape(concept_ind_y, -1)
        num_vocab = np.shape(concept_ind_x)[1]
        new_vocabulary_words = ["{} {}".format(vocabulary_list[concept_ind_x[0, ii]], vocabulary_list[concept_ind_y[0, ii]]) for ii in range(0, num_vocab)]
        new_vocabulary_frequency = [comatrix[concept_ind_x[0, ii], concept_ind_y[0, ii]] for ii in range(0, num_vocab)]
        new_vocabulary_pmi = [co_pmi[concept_ind_x[0, ii], concept_ind_y[0, ii]] for ii in range(0, num_vocab)]

        new_vocabulary = zip(new_vocabulary_words, new_vocabulary_frequency, new_vocabulary_pmi)
        #Sort new_vocabulary from highest PMI to lowest
        new_vocabulary.sort(key=lambda x: x[2], reverse=True)
    else:
        print("PMI error")
        new_vocabulary = []

    return new_vocabulary

if __name__ == "__main__":
    comatrix_path = 'E:\\rooms\\structure\\rooms_200_extended_flickr_comatrix_approx.mat'
    pmi_threshold = 0.5
    image_threshold = 200
    countstr = 'image'

    concept_file = 'E:\\rooms\\data\\concepts\\rooms_200_extended_split_owner_per_concept.txt'
    with open(concept_file, 'r') as f:
        all_concepts = [(x[0], int(float(x[1]))) for x in [t.split('\t') for t in f.read().split('\n')]]
    num_vocabulary = len(all_concepts)
    vocabulary_list, vocabulary_freq = zip(*all_concepts)

    additional_mask = np.tril(np.ones((num_vocabulary,num_vocabulary)), -1)
    high_pmi(comatrix_path, pmi_threshold, image_threshold, countstr, additional_mask, vocabulary_list)

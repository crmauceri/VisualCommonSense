__author__ = 'Cecilia'
import gensim
import numpy as np
import scipy.io as spio
from sklearn.cluster import MiniBatchKMeans as kmeans
import os


def get_word_2_vec(model_file, save_file, concept_file):
    model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=True)

    with open(concept_file, 'r') as f:
        concepts = f.read().split('\n')
        concepts = [c.split('\t')[0].replace(' ', '_') for c in concepts if len(c) > 0]

    features = np.Inf * np.ones((model.layer1_size, len(concepts)))
    mean_feature_mask = np.zeros((len(concepts),))
    feature_mask = np.ones((len(concepts),))

    for i in range(0, len(concepts)):
        if concepts[i] in model:
            features[:, i] = model[concepts[i]]
        elif concepts[i].replace('_', '') in model:
            features[:, i] = model[concepts[i].replace('_', '')]
        elif all([word in model for word in concepts[i].split('_')]):
            #If the phrase is not in the glove dictionary, but the component words are, use the mean of the vectors
            mean_feature_mask[i] = 1
            concept_words = concepts[i].split('_')
            features[:, i] = 0
            for word in concept_words:
                features[:, i] = features[:, i] + model[word]
            features[:, i] = features[:, i] / len(concept_words)
        else:
            feature_mask[i] = 0
            print "{} not in model".format(concepts[i])

    features = features.transpose()
    spio.savemat(save_feature_file, {'features': features, 'concepts': concepts, 'feature_mask': feature_mask, 'mean_feature_mask': mean_feature_mask})


# Adapted from dhammacks Word2VecExample
def load_glove_vec(txt_filepath, num_dims):
    glove_terms = []
    vocab_ind = 0
    with open(txt_filepath, 'r') as fin:
        for line in fin:
            items = line.replace('\r', '').replace('\n', '').split(' ')
            if len(items) < 10: continue
            vocab_ind += 1

    glove_vec = np.zeros((vocab_ind, num_dims))
    vocab_ind = 0
    # load the word2vec features.
    with open(txt_filepath, 'r') as fin:
        # if path == 'vectors0.txt':
        # next(fin) #skip information on first line
        for line in fin:
            items = line.replace('\r', '').replace('\n', '').split(' ')
            if len(items) < 10: continue
            glove_terms.append(items[0])
            vect = np.array([float(i) for i in items[1:] if len(i) > 0])
            glove_vec[vocab_ind, :] = vect
            vocab_ind += 1

    return glove_vec, glove_terms

def get_glove(dim, model_file, save_model_file, save_feature_file, concept_file):
    if os.path.exists(save_model_file):
        model_mat = spio.loadmat(save_model_file)
        glove_vec = model_mat['glove_vec']
        glove_terms = model_mat['glove_terms']
    else:
        print "loading {}".format(model_file)
        glove_vec, glove_terms = load_glove_vec(model_file, dim)
        print "loaded"

        #spio.savemat(save_model_file, {'glove_vec': glove_vec, 'glove_terms': glove_terms})
        #print "saved"

    with open(concept_file, 'r') as f:
        concepts = f.read().split('\n')
        concepts = [c.split('\t')[0].replace(' ', '_') for c in concepts]

    num_concepts = len(concepts)
    features = np.Inf * np.ones((num_concepts, dim))
    mean_feature_mask = np.zeros((num_concepts,))
    feature_mask = np.ones((num_concepts,))
    glove_terms_dict = dict(zip(glove_terms, range(0, len(glove_terms))))

    for i in range(0, num_concepts):
        if concepts[i] in glove_terms_dict:
            features[i, :] = glove_vec[glove_terms_dict[concepts[i]], :]
        elif concepts[i].replace('_', '') in glove_terms_dict:
            features[i, :] = glove_vec[glove_terms_dict[concepts[i].replace('_', '')], :]
        elif all([word in glove_terms_dict for word in concepts[i].split('_')]):
            #If the phrase is not in the glove dictionary, but the component words are, use the mean of the vectors
            mean_feature_mask[i] = 1
            concept_words = concepts[i].split('_')
            features[i, :] = 0
            for word in concept_words:
                features[i, :] = features[i, :] + glove_vec[glove_terms_dict[word], :]
            features[i, :] = features[i, :] / len(concept_words)
        else:
            feature_mask[i] = 0
            print "{} not in model".format(concepts[i])

    spio.savemat(save_feature_file, {'features': features, 'concepts': concepts, 'feature_mask': feature_mask, 'mean_feature_mask': mean_feature_mask})


if __name__ == "__main__":
    query = "biology_domesticated_animal"
    min_num_images = 500
    concept_file = 'E:\data\Iconic\data\\test_crawler\data\concepts\\{}_{}_extended_owner_per_concept.txt'.format(query, min_num_images)

    model_file = 'E:\data\GloVe\glove.42B.300d.txt'
    save_model_file = ''
    dim = 300
    save_feature_file = "E:\data\Iconic\data\word2vec_features\\{}_{}_extended_feature_glove.42B.300d.mat".format(query, min_num_images)
    #get_glove(dim, model_file, save_model_file, save_feature_file, concept_file)

    model_file = 'E:\data\word2vec\GoogleNews-vectors-negative300.bin.gz'
    save_file = "E:\data\Iconic\data\word2vec_features\\{}_{}_extended_feature_word2vec.mat".format(query, min_num_images)
    get_word_2_vec(model_file, save_file, concept_file)
__author__ = 'mauceri2'
from database_builder.core.timing import tic
from database_builder.core.timing import toc
from database_builder.tools.cmd_arguments_helper import CmdArgumentsHelper
import os, math, re
import numpy as np
from numpy import inf
import crawler.config as cfg
import scipy.io as spio
from itertools import permutations

def main():
    arg_helper = CmdArgumentsHelper()
    arg_helper.add_argument('query', 'q', 'query', 1)
    arg_helper.add_argument('root_dir', 'r', 'root', 1)
    arg_helper.add_argument('stats_dir', 's', 'stats', 1)
    arg_helper.add_argument('knowledge_dir', 'k', 'knowledge', 1)
    arg_helper.add_argument('min_num_images', 'n', 'min_num_images', 1)
    args = arg_helper.read_arguments()
    print (args)

    root = args['root_dir']
    min_num_images = int(args['min_num_images'])
    query = args['query']
    stats_dir = args['stats_dir']
    knowledge_dir = args['knowledge_dir']

    # Config file configuration stuff
    argv = cfg.vars
    numberOfThreads = int(argv["numberOfThreads"])

    save_description = "{}_{}".format(query, min_num_images)

    # If do_skip is set to 1, will skip the item that we have previously
    # generated data. You should be careful to set it as 1, unless you
    # are certain the data will not change, i.e. if you remove or add new
    # photo to the image id list, you should set it as 0 to regenerate
    # all related data.
    do_skip = True


    # Find most frequent concepts
    # Concept List will be saved in file
    concept_dir = os.path.join(root, "concepts")
    if not os.path.exists(concept_dir):
        os.mkdir(concept_dir)

    concept_file = os.path.join(concept_dir, '{}_owner_per_concept.txt'.format(save_description))

    from database_builder.get_photo_meta_multiprocess import find_vocabulary
    tic()
    find_vocabulary(root, stats_dir, query, min_num_images, save_description)
    toc()

    with open(concept_file, 'r') as f:
        all_concepts = all_concepts = [(x[0], int(float(x[1]))) for x in [t.split('\t') for t in f.read().split('\n')]]

    all_concepts_list,  all_concepts_freq = zip(*all_concepts)
    spio.savemat(concept_file[:-3] + 'mat', {'concepts': all_concepts_list})

    #Remove some of the vocabulary fitting certain criteria
    filter_vocab_dir = os.path.join(root, 'filter_lists')
    all_concepts = filter_vocabulary(filter_vocab_dir, all_concepts)

    save_description = "{}_{}_extended".format(query, min_num_images)
    concept_file = os.path.join(concept_dir, '{}_filtered_owner_per_concept.txt'.format(save_description))
    with open(concept_file, 'w') as f:
        all_concepts_str = ["{}\t{}".format(t[0], t[1]) for t in all_concepts]
        f.write("\n".join(all_concepts_str))

    with open(concept_file, 'r') as f:
        all_concepts = [(x[0], int(float(x[1]))) for x in [t.split('\t') for t in f.read().split('\n')]]

    #Break concatenated word pairs
    all_concepts = break_pairs(all_concepts)

    concept_file = os.path.join(concept_dir, '{0}_split_owner_per_concept.txt'.format(save_description))
    with open(concept_file, 'w') as f:
        all_concepts_str = ["{}\t{}".format(t[0], t[1]) for t in all_concepts]
        f.write("\n".join(all_concepts_str))

    with open(concept_file, 'r') as f:
        all_concepts = [(x[0], int(float(x[1]))) for x in [t.split('\t') for t in f.read().split('\n')]]

    #Find approximate statistics for concept pairs
    #For initial vocabulary expansion
    web_dir = os.path.join(root, 'output')
    if not os.path.exists(web_dir):
        os.mkdir(web_dir)
    from database_builder.get_photo_meta_multiprocess import find_approximate_concept_pairs
    tic()
    find_approximate_concept_pairs(root, stats_dir, query, save_description, all_concepts)
    toc()

    # Expand vocabulary
    all_concepts = extend_vocabulary(root, stats_dir, knowledge_dir, all_concepts, query, min_num_images, save_description, do_skip)

    concept_file = os.path.join(concept_dir, '{}_owner_per_concept.txt'.format(save_description))
    with open(concept_file, 'w') as f:
        all_concepts_str = ["{}\t{}".format(t[0], t[1]) for t in all_concepts]
        f.write("\n".join(all_concepts_str))

    with open(concept_file, 'r') as f:
        all_concepts = [(x[0], int(float(x[1]))) for x in [t.split('\t') for t in f.read().split('\n')] if len(x)>1]

    all_concepts = merge_pairs(all_concepts)
    all_concepts_list, all_concepts_freq = zip(*all_concepts)

    spio.savemat(concept_file[:-3] + 'mat', {'concepts': all_concepts_list})

    #Recount tag cooccurrence with final vocabulary
    from database_builder.get_photo_meta_multiprocess import find_concept_pairs
    #if total_new_concepts > 0:
    tic()
    web_dir = os.path.join(root, 'output')
    if not os.path.exists(web_dir):
        os.mkdir(web_dir)
    find_concept_pairs(root, stats_dir, web_dir, query, all_concepts)
    toc()

    #Process knowledge
    from structured_knowledge.parse_cache import parse_cache
    from structured_knowledge.download_structured_knowledge import download_structured_knowledge
    download_structured_knowledge(knowledge_dir, all_concepts_list, do_skip)
    parse_cache(knowledge_dir, "ConceptNet", all_concepts, save_description, stats_dir)
    parse_cache(knowledge_dir, "Freebase", all_concepts, save_description, stats_dir)

    #Generate adjacency matrices
    from structured_knowledge.build_adjacency_matrices import build_adjacency_matrices
    build_adjacency_matrices(knowledge_dir, stats_dir, all_concepts_list, save_description)

    from database_builder.gen_concept_structure import task_gen_synonym_mask
    from database_builder.gen_concept_structure import task_gen_lemma_mask

    task_gen_synonym_mask(all_concepts_list, stats_dir, save_description)
    task_gen_lemma_mask(all_concepts_list, stats_dir, save_description)

    from structured_knowledge.parts_of_speech import parse_language
    from structured_knowledge.parts_of_speech import parse_proper_nouns
    from structured_knowledge.parts_of_speech import parse_parts_of_speech
    from structured_knowledge.parse_object_scene import parse_object_concepts
    from structured_knowledge.parse_object_scene import parse_scene_concepts

    parse_language(all_concepts_list, stats_dir, save_description)
    parse_proper_nouns(all_concepts_list, stats_dir, save_description)
    parse_parts_of_speech(all_concepts_list, knowledge_dir, stats_dir, save_description)
    parse_scene_concepts(knowledge_dir, stats_dir, all_concepts_list, save_description)
    parse_object_concepts(knowledge_dir, stats_dir, all_concepts_list, save_description)

    gen_phrase_mask(all_concepts_list, stats_dir, save_description)

    from database_builder.get_vocab_features import get_glove
    print("Start GloVe Feature")
    model_file = 'E:\data\GloVe\glove.42B.300d.txt'
    save_model_file = ''
    dim = 300
    save_feature_file = "E:\data\Iconic\data\word2vec_features\\{}_extended_feature_glove.42B.300d.mat".format(save_description)
    get_glove(dim, model_file, save_model_file, save_feature_file, concept_file)


def gen_phrase_mask(all_concepts_list, stats_dir, save_description):
    num_vocabulary = len(all_concepts_list)
    component_vocab_mat = np.zeros((num_vocabulary, num_vocabulary))
    all_concepts_dict = dict(list(zip(all_concepts_list, list(range(0, num_vocabulary)))))
    for phrase in all_concepts_list:
        for word in phrase.split():
            if word in all_concepts_dict:
                component_vocab_mat[all_concepts_dict[word], all_concepts_dict[phrase]] = 1
                component_vocab_mat[all_concepts_dict[phrase], all_concepts_dict[word]] = 1
        if all([word in all_concepts_dict for word in phrase.split()]):
            combo = permutations(phrase.split(), 2)
            for c in combo:
                component_vocab_mat[all_concepts_dict[c[0]], all_concepts_dict[c[1]]] = 1

    save_path = os.path.join(stats_dir, save_description + '_phrase_mask.mat')
    spio.savemat(save_path, {'phrase_mask': component_vocab_mat})



def filter_vocabulary(filter_vocab_dir, vocabulary):
    vocabulary_list, vocabulary_freq = list(zip(*vocabulary))
    vocabulary_dict = dict(list(zip(vocabulary_list, list(range(0, len(vocabulary))))))

    #Remove vocabulary with fewer than 3 characters or which is an english stopword
    from nltk.corpus import stopwords
    eng_stopwords = stopwords.words('english')
    new_vocabulary_list = [w for w in vocabulary_list if (not w in eng_stopwords) and len(w) > 2]

    #Remove vocabulary terms meeting the following criteria
    #Camera vocabulary
    with open(os.path.join(filter_vocab_dir, 'digital_camera_vocab.txt')) as f:
        camera_vocab_set = set([w.strip().lower() for w in f.read().split('\n')])
    new_vocabulary_list = [w for w in new_vocabulary_list if not w in camera_vocab_set]

    #Regex "photo"
    photo_pattern = re.compile('photo')
    new_vocabulary_list = [w for w in new_vocabulary_list if photo_pattern.search(w) is None]

    #Numbers
    num_pattern = re.compile('\d+')
    new_vocabulary_list = [w for w in new_vocabulary_list if num_pattern.search(w) is None]

    #Ambiguous vocab i.e. strings containing only unrecognised characters
    ambig_pattern = re.compile('^[?]+$')
    new_vocabulary_list = [w for w in new_vocabulary_list if ambig_pattern.match(w) is None]

    #Auto tags from Flickr
    auto_pattern = re.compile(':')
    vision_pattern = re.compile('vision')
    new_vocabulary_list = [w for w in new_vocabulary_list if (auto_pattern.search(w) is None) or (not vision_pattern.search(w) is None)]

    new_vocabulary_freq = [vocabulary_freq[vocabulary_dict[w]] for w in new_vocabulary_list]
    return list(zip(new_vocabulary_list, new_vocabulary_freq))


def extend_vocabulary(root, stats_dir, knowledge_dir, freq_vocabulary, query, min_num_images, save_description, do_skip):
    num_vocabulary = len(freq_vocabulary)
    freq_vocabulary_list = [t[0] for t in freq_vocabulary]
    freq_vocabulary_nospace = ["".join(w.split()) for w in freq_vocabulary_list]
    freq_vocabulary_dict = dict(list(zip(freq_vocabulary_list + freq_vocabulary_nospace, list(range(0, len(freq_vocabulary_list)))+list(range(0, len(freq_vocabulary_list))))))
    freq_vocabulary_set = set(freq_vocabulary_list)

    #Check adjacent Freebase concepts for related vocabulary phrases
    #----------------------------------------------------------------
    from structured_knowledge.download_structured_knowledge import download_structured_knowledge
    from structured_knowledge.parse_cache import find_candidates
    download_structured_knowledge(knowledge_dir, freq_vocabulary_list, do_skip)
    freq_vocabulary_edit, candidate_vocabulary_list, errors = find_candidates(knowledge_dir, "Freebase", freq_vocabulary, save_description)

    #Find the frequency of two word phrases using co-occurrence matrix
    comatrix_path = os.path.join(stats_dir, save_description + '_flickr_comatrix_approx.mat')
    comatrix_dict = spio.loadmat(comatrix_path)
    comatrix = comatrix_dict['comatrix_images']
    comatrix[np.eye(np.shape(comatrix)[0], dtype=bool)] = 0 #Mask diagonal

    two_word_phrases = [w for w in candidate_vocabulary_list if len(w.split())==2]
    two_word_indices = [(freq_vocabulary_dict[t[0]], freq_vocabulary_dict[t[1]]) for t in (w.split() for w in two_word_phrases)]
    confirmed_new_vocab_pair = [(two_word_phrases[i], comatrix[two_word_indices[i]]) for i in range(0, len(two_word_phrases)) if (comatrix[two_word_indices[i]] > min_num_images) or ("".join(two_word_phrases[i].split()) in freq_vocabulary_set)]

    #Longer phrases remain in the candidate_vocabulary
    candidate_vocabulary = [(w, 0) for w in candidate_vocabulary_list if len(w.split()) > 2]

    #Count candidate frequency
    from database_builder.get_photo_meta_multiprocess import freq_pattern_count
    confirmed_new_vocab_long = freq_pattern_count(knowledge_dir, root, stats_dir, query, save_description, candidate_vocabulary, min_num_images)
    #confirmed_new_vocab_long = [('hard rock cafe', 262.0), ('empire state building', 221.0)]

    #Add concept pairs with high PMI to concepts list
    #----------------------------------------------------------------
    from analysis.python_pmi import high_pmi
    tic()
    score_threshold = 0.8
    new_vocab = high_pmi(comatrix_path, score_threshold, min_num_images, 'image', np.tril(np.ones((num_vocabulary,num_vocabulary)), -1), freq_vocabulary_list)

    #Exclude vocab that has already been discovered using Freebase
    vocab_split_set = set(zip(*freq_vocabulary_edit)[2])
    new_vocab = [w for w in new_vocab if not w[0] in two_word_phrases and not w[0] in vocab_split_set]
    new_vocab_list = [t[0] for t in new_vocab]
    new_vocab_set = set(new_vocab_list)

    #Don't use more that num_extra_terms new vocab. This is to limit the number of irrelevant terms downloaded
    num_extra_terms = 500
    new_vocab = new_vocab[:min(len(new_vocab), num_extra_terms)]
    new_vocab_list = [t[0] for t in new_vocab]

    from structured_knowledge.parse_cache import vet_candidates
    download_structured_knowledge(knowledge_dir, new_vocab_list, do_skip)
    confirmed_new_vocab_pmi = vet_candidates(knowledge_dir, "Freebase", new_vocab)
    toc()

    # Extend vocabulary and parse Knowledgebase with complete vocabulary
    #--------------------------------------------------------------------
    confirmed_new_vocab_pair = [(t[0], t[1], "".join(t[0].split())) for t in confirmed_new_vocab_pair]
    confirmed_new_vocab_long = [(t[0], t[1], "".join(t[0].split())) for t in confirmed_new_vocab_long]
    confirmed_new_vocab_pmi = [(t[0], t[1], "".join(t[0].split())) for t in confirmed_new_vocab_pmi]
    complete_vocabulary = freq_vocabulary_edit + confirmed_new_vocab_pair + confirmed_new_vocab_long + confirmed_new_vocab_pmi

    complete_vocabulary_dict = {}
    for tup in complete_vocabulary:
        if tup[2] in complete_vocabulary_dict:
            if tup[0] != complete_vocabulary_dict[tup[2]]:
                complete_vocabulary_dict[tup[2]] = (tup[0], tup[1] + complete_vocabulary_dict[tup[2]][1])
        else:
            complete_vocabulary_dict[tup[2]] = (tup[0], tup[1])

    complete_vocabulary = complete_vocabulary_dict.values()
    complete_vocabulary.sort(key=lambda x: x[1], reverse=True)
    return complete_vocabulary


def break_pairs(freq_vocabulary):
    #Use all combinations to break words that have been concatenated without spaces
    freq_vocabulary_list, freq_vocabulary_freq = list(zip(*freq_vocabulary))
    freq_vocabulary_list = list(freq_vocabulary_list)
    freq_vocabulary_set = set(freq_vocabulary_list)
    num_vocabulary = len(freq_vocabulary_list)
    freq_vocabulary_dict = dict(list(zip(freq_vocabulary_list, list(range(0,num_vocabulary)))))

    #First use Wordnet to identify common dictionary words
    from nltk.corpus import wordnet as wn
    common = set([w for w in freq_vocabulary_list if len(wn.synsets(w)) > 0])
    common_index = [freq_vocabulary_dict[w] for w in common]

    from itertools import permutations
    new_vocab = {}
    for pair in permutations(range(0, num_vocabulary), 2):
        word1 = freq_vocabulary_list[pair[0]]
        word2 = freq_vocabulary_list[pair[1]]
        freq_estimate = freq_vocabulary_freq[pair[0]] + freq_vocabulary_freq[pair[1]]
        concept = '{}{}'.format(word1, word2)
        if (concept in freq_vocabulary_set) and (not concept in common):
            if concept in new_vocab:
                #If  there is more than one way to split a concept, take the one with the more frequent components
                if freq_estimate > new_vocab[concept][1]:
                    new_vocab[concept] = ("{} {}".format(word1, word2), freq_estimate)
            else:
                new_vocab[concept] = ("{} {}".format(word1, word2), freq_estimate)

    for concept in new_vocab:
        word_ind = freq_vocabulary_dict[concept]
        freq_vocabulary_list[word_ind] = new_vocab[concept][0]

    new_freq_vocabulary = list(zip(freq_vocabulary_list, freq_vocabulary_freq))

    return new_freq_vocabulary


def merge_pairs(freq_vocabulary):
    #If breaking concatenated words using Freebase aliases was too aggressive, merge words which are in Wordnet
    freq_vocabulary_list, freq_vocabulary_freq = zip(*freq_vocabulary)
    freq_vocabulary_list = list(freq_vocabulary_list)
    num_vocabulary = len(freq_vocabulary_list)

    from nltk.corpus import wordnet as wn
    for ii in range(0, num_vocabulary):
        concept = freq_vocabulary_list[ii]
        if len(concept.split()) > 1:
            concept_no_space = concept.replace(' ', '')
            if (len(wn.synsets(concept_no_space)) > 0) and (len(wn.synsets(concept.replace(' ', '_'))) == 0):
                freq_vocabulary_list[ii] = concept_no_space

    return zip(freq_vocabulary_list, freq_vocabulary_freq)



if __name__ == '__main__':
    main()
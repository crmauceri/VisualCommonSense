__author__ = 'mauceri2'
import numpy as np
import scipy.io as sp
import scipy.stats as stat
import itertools as it
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import json
import os

wnl = WordNetLemmatizer()
pst = PorterStemmer()

def task_gen_lemma_mask(concept_list, statsroot, search_description):
    num_concepts = len(concept_list)
    shape = [num_concepts, num_concepts]

    #WordNet lemmatizer
    wnllemmas = [[wnl.lemmatize(concept), concept] for concept in concept_list]
    wnllemmas = [c for c in wnllemmas if c[0] in concept_list and not c[0] == c[1]]
    wnlLemmaIndices = [(concept_list.index(c[0]), concept_list.index(c[1])) for c in wnllemmas]
    
    #Porter Stemmer
    pstlemmas = [[pst.stem(concept), concept] for concept in concept_list]
    pstlemmas = [c for c in pstlemmas if c[0] in concept_list and not c[0] == c[1]]
    pstLemmaIndices = [(concept_list.index(c[0]), concept_list.index(c[1])) for c in pstlemmas]

    lemmaIndices = list(set(wnlLemmaIndices + pstLemmaIndices))
    mask = np.zeros(shape)
    for lemma in lemmaIndices:
        #print "{}: {} {}".format(lemma, concept_list[lemma[1]], concept_list[lemma[0]])
        mask[lemma[1], lemma[0]] = 1
    
    path = os.path.join(statsroot, "{}_lemma_mask.mat".format(search_description))
    sp.savemat(path, {'lemma_mask': mask})

    print "done generating lemma mask"
    return mask

def task_gen_synonym_mask(concept_list, statsroot, search_description):
    num_concepts = len(concept_list)
    shape = [num_concepts, num_concepts]
    synonym_mask = np.zeros(shape)

    for concept in concept_list:
        synonyms = []
        for synset in wn.synsets(concept):
            lemmas = [str(lemma.name()) for lemma in synset.lemmas()]
            synonyms += [lemma for lemma in lemmas if lemma in concept_list]

        synonyms = list(set(synonyms))
        synonyms_ind = [concept_list.index(s) for s in synonyms]
        synonyms_ind.sort()

        synonym_mask[concept_list.index(concept), synonyms_ind] = 1

    path = os.path.join(statsroot, "{}_synonym_mask.mat".format(search_description))
    sp.savemat(path, {'synonym_mask': synonym_mask})

    print "done generating synonym mask"
    return synonym_mask

def task_gen_tag_stats(photos, wordlist, patternlist, statsroot, webroot, search_description, conceptKeyList):
    #Generate features for patterns
    num_words = len(wordlist)
    num_patterns = len(patternlist)
    freq_pattern_feature = np.zeros((num_patterns, num_words))
    num_words_in_pattern = np.zeros((num_patterns, 1))
    for patternInd in range(0, num_patterns):
        pattern = patternlist[patternInd]
        pattern_delim = pattern.split()
        wordInd = [wordlist.index(p) for p in pattern_delim]
        freq_pattern_feature[patternInd, wordInd] = 1
        num_words_in_pattern[patternInd] = len(pattern_delim)

    image_count = 0
    total_pattern_count = []
    pattern_count = np.zeros([num_patterns, 1]) #vector of occurances of each concept
    co_matrix = np.zeros([num_patterns, num_patterns]) #matrix of occurances of each pair of concepts
    owners_matrix = np.zeros([num_patterns, num_patterns]) #matrix of occurances of each pair of concepts by a unique owner
    owners_by_pattern_pair = {} #dict of owners who use each tag pair
    images_by_pattern_pair = {}
    images_by_pattern = {}
    pattern_owners = {} #dict of owners who use each tag
    owners_set = set([]) #list of all owners

    for photo in photos:
        concepts = []
        for conceptKey in conceptKeyList:
            if conceptKey in photo:
                concepts += photo[conceptKey]

        concepts = list(set([c for c in concepts if c in wordlist]))
        if len(concepts) > 0:
            concept_indices = [wordlist.index(c) for c in concepts]
            concept_indices = list(set(concept_indices))
            concept_feature = np.zeros((num_words, 1))
            concept_feature[concept_indices] = 1

            freq_pattern_count = np.zeros([num_patterns, 1])
            freq_pattern_count[np.dot(freq_pattern_feature, concept_feature) == num_words_in_pattern] += 1

            if np.sum(freq_pattern_count)>1:
                owner = photo["owner"]
                owners_set.add(owner)

                pattern_indices = np.where(freq_pattern_count==1)[0]
                for p in pattern_indices:
                    if not p in pattern_owners:
                        #New tag
                        pattern_owners[p] = set([owner])
                    else:
                        pattern_owners[p].add(owner)

                    if not p in images_by_pattern:
                        images_by_pattern[patternlist[p]] = [photo["photo"]]
                    else:
                        images_by_pattern[patternlist[p]].append(photo["photo"])
                    pattern_count[p] += 1

                image_count += 1
                total_pattern_count.append(len(pattern_indices))

                for combo in it.combinations(pattern_indices, 2):
                    co_matrix[combo[0],combo[1]] += 1
                    co_matrix[combo[1],combo[0]] += 1

                    if not combo in owners_by_pattern_pair:
                        #New tag pair
                        owners_by_pattern_pair[combo] = [owner]
                        owners_matrix[combo[0], combo[1]] += 1
                    else:
                        if not owner in owners_by_pattern_pair[combo]:
                            #New owner
                            owners_matrix[combo[0], combo[1]] += 1
                            owners_by_pattern_pair[combo].append(owner)

                    pattern0 = patternlist[combo[1]]
                    pattern1 = patternlist[combo[0]]
                    if not pattern0 in images_by_pattern_pair:
                        images_by_pattern_pair[pattern0] = {pattern1: [photo["photo"]]}
                    elif not pattern1 in images_by_pattern_pair[pattern0]:
                        images_by_pattern_pair[pattern0][pattern1] = [photo["photo"]]
                    else:
                        images_by_pattern_pair[pattern0][pattern1].append(photo["photo"])

    pattern_owner_count = np.zeros([num_patterns, 1])
    for s in pattern_owners:
        pattern_owner_count[s] = len(pattern_owners[s])

    num_owners = len(owners_set)

    path = os.path.join(statsroot, "{}_owners.mat".format(search_description))
    sp.savemat(path, {'owners_matrix': owners_matrix, 'pattern_owner_count': pattern_owner_count, 'total_owners': num_owners})

    path = os.path.join(statsroot, "{}_flickr.mat".format(search_description))
    sp.savemat(path, {'comatrix': co_matrix, 'image_count': image_count, 'tag_count': total_pattern_count})

    out_dir = os.path.join(webroot, search_description)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for tag in images_by_pattern_pair:
        pair_dir = os.path.join(out_dir, "pair")
        if not os.path.exists(pair_dir):
            os.makedirs(pair_dir)

        path = os.path.join(pair_dir, "{}.json".format(tag))
        with open(path, 'w') as fin:
            json.dump(images_by_pattern_pair[tag], fin)

    for tag in images_by_pattern:
        single_dir = os.path.join(out_dir, "concepts")
        if not os.path.exists(single_dir):
            os.makedirs(single_dir)

        path = os.path.join(single_dir, "{}.json".format(tag))
        with open(path, 'w') as fin:
            json.dump(images_by_pattern[tag], fin)

    #Report Stats
    path = os.path.join(statsroot, "{}_stats.txt".format(search_description))
    with open(path, 'w') as fin:
        fin.write('Number of images: {}\n'.format(image_count))
        fin.write('Number of owners: {}\n'.format(num_owners))
        fin.write('Mean number of concepts: {}\n'.format(np.mean(total_pattern_count)))
        fin.write('Median number of concepts: {}\n'.format(np.median(total_pattern_count)))
        fin.write('Q1 number of concepts: {}\n'.format(np.percentile(total_pattern_count, 25)))
        fin.write('Q3 number of concepts: {}\n'.format(np.percentile(total_pattern_count, 75)))
        fin.write('Std number of concepts: {}\n'.format(np.std(total_pattern_count)))
        fin.write('Max number of concepts: {}\n'.format(max(total_pattern_count)))
        fin.write('Min number of concepts: {}'.format(min(total_pattern_count)))

    path = os.path.join(statsroot, "{}_stats.mat".format(search_description))
    sp.savemat(path, {'num_concepts': np.matrix(total_pattern_count)})

    path = os.path.join(statsroot, "{}_concepts.mat".format(search_description))
    sp.savemat(path, {'concepts': patternlist})

    print "Done generating coocurrance matrices"


if __name__ == '__main__':
    import sys, os, csv
    if len(sys.argv)<7:
        print 'Too few arguments. Execute as >> python gen_concepts_structure.py root statsroot webroot query conceptKeyList numConcepts'
    from database_builder.tools.query_descriptor import query_descriptor
    search_description = query_descriptor(sys.argv[4], int(sys.argv[6]), [sys.argv[5]])

    from get_photo_meta import get_concept_frequency
    concept_path = os.path.join(sys.argv[1], 'data', 'concepts')
    concept_list, scores = get_concept_frequency(concept_path, sys.argv[4], int(sys.argv[6]), [sys.argv[5]], 'all_concepts')

    task_gen_lemma_mask(concept_list, sys.argv[2], search_description)
    # task_gen_synonym_mask(concept_list, sys.argv[2], search_description)
    #
    # pattern_list = get_concept_list(concept_path, sys.argv[4], int(sys.argv[6]), [sys.argv[5]], 'all_concepts')
    #
    # from get_photo_meta import get_photo_meta
    # photos = get_photo_meta(sys.argv[1], sys.argv[4])
    #
    # task_gen_tag_stats(photos, concept_list, pattern_list, sys.argv[2], sys.argv[3], search_description, [sys.argv[5]])
    print "finished"
__author__ = 'mauceri2'

import os
import numpy as np
from numpy import ix_
import scipy.io as sp
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
import collections

ps = PorterStemmer()
wnl = WordNetLemmatizer()

POS = 'NVAvCP!rDIo'

def parse_language(concept_list, save_dir, search_descriptor):
    print("begin parse languages")

    num_concepts = len(concept_list)
    languages =['als', 'arb', 'cmn', 'dan', 'eng', 'fas', 'fin',
        'fra', 'fre', 'heb', 'ita', 'jpn', 'spa', 'pol', 'por', 'tha']

    num_lang = len(languages)
    is_language = np.zeros((num_concepts, num_lang))
    is_translation = np.zeros((num_concepts, num_concepts))

    concept_synsets = collections.defaultdict(set)
    #First pass through collect all synsets
    for concept_ind in range(0, num_concepts):
        concept_underscore = concept_list[concept_ind].replace(" ", "_")
        synsets = wn.synsets(concept_underscore)
        for synset in synsets:
            concept_synsets[synset.name()].add(concept_ind)

        for language_ind in range(0, num_lang):
            cur_lang = languages[language_ind]
            synsets = wn.synsets(concept_underscore, lang=cur_lang)

            if len(synsets) > 0:
                is_language[concept_ind, language_ind] = 1

            for synset in synsets:
                concept_synsets[synset.name()].add(concept_ind)

    for synset in concept_synsets:
        cur_concept_ind = list(concept_synsets[synset])
        is_translation[ix_(cur_concept_ind, cur_concept_ind)] = 1

    save_path = os.path.join(save_dir, search_descriptor+'_languages.mat')
    sp.savemat(save_path, {"is_language":is_language, "language_ISO":languages, "is_translation": is_translation, "concepts": concept_list})

    print("done parse language")


def parse_proper_nouns(concept_list, save_dir, search_descriptor):
    print("begin parse proper nouns")

    num_vocabulary = len(concept_list)
    vocabulary_dict = dict(zip(concept_list, range(0, num_vocabulary)))
    proper_noun_vector = np.zeros((num_vocabulary,))
    common_noun_vector = np.zeros((num_vocabulary,))
    english_noun_vector = np.zeros((num_vocabulary))

    for concept in concept_list:
        concept_underscore = concept.replace(" ", "_")
        has_english_meaning = len(wn.synsets(concept_underscore, lang="eng"))>0
        has_common_meaning = has_english_meaning
        has_proper_meaning = False
        for synset in wn.synsets(concept_underscore):
            has_proper_meaning = has_proper_meaning or len(synset.instance_hypernyms()) > 0
            has_common_meaning = has_common_meaning or len(synset.hypernyms()) > 0

        if has_proper_meaning and not has_common_meaning:
            proper_noun_vector[vocabulary_dict[concept]] = 1
        if has_common_meaning:
            common_noun_vector[vocabulary_dict[concept]] = 1
        if has_english_meaning:
            english_noun_vector[vocabulary_dict[concept]] = 1

    save_path = os.path.join(save_dir, search_descriptor+'_proper_nouns.mat')
    sp.savemat(save_path, {"proper_nouns": proper_noun_vector, "common_nouns": common_noun_vector, "english_nouns": english_noun_vector, "concepts": concept_list})

    print("done parse proper nouns")


# Some Moby labels are too specific, so replace them with more general variant
#     Noun                            N
#     Plural                          p
#     Noun Phrase                     h
#     Verb (usu participle)           V
#     Verb (transitive)               t
#     Verb (intransitive)             i
#     Adjective                       A
#     Adverb                          v
#     Conjunction                     C
#     Preposition                     P
#     Interjection                   !
#     Pronoun                         r
#     Definite Article                D
#     Indefinite Article              I
#     Nominative                      o
def pos_replace(s):
    replace_dict = {'p': 'N', 'h': 'N', 't': 'V', 'i': 'V'}
    new_string = ""
    for c in s:
        if c in replace_dict:
            new_string += replace_dict[c]
        else:
            new_string += c
    return new_string

def parse_parts_of_speech(concept_list, knowledge_dir, save_dir, search_descriptor):
    print("start parse parts of speech")
    pos = list(POS)
    pos_path = os.path.join(knowledge_dir, 'mpos', 'mobyposi.i')
    with open(pos_path, 'r') as f:
        pos_text = f.read().split('\r')
    pos_list = [p.split('\xd7') for p in pos_text]
    pos_dict = {p[0].lower().decode("ascii", errors="replace").encode('ascii', errors='replace'): p[1] for p in pos_list if len(p) > 1}
    # pos_dict_stem = {}
    # for k in pos_dict.keys():
    #     pos_dict_stem[ps.stem(k)] = k
    #     pos_dict_stem[wnl.lemmatize(k)] = k

    concept_pos_mat = np.zeros([len(concept_list), len(pos)])
    concept_moby_mat = np.zeros([len(concept_list), len(pos)])
    no_pos = []
    for concept in concept_list:
        concept_underscore = concept.replace(" ", "_")
        present = False

        #Check WordNet part of speech
        if len(wn.synsets(concept_underscore, pos=wn.VERB)) > 0:
            concept_pos_mat[concept_list.index(concept), pos.index('V')] = 1
            present = True
        if len(wn.synsets(concept_underscore, pos=wn.NOUN)) > 0:
            concept_pos_mat[concept_list.index(concept), pos.index('N')] = 1
            present = True
        if len(wn.synsets(concept_underscore, pos=wn.ADJ)) > 0:
            concept_pos_mat[concept_list.index(concept), pos.index('A')] = 1
            present = True

        if not present and len(concept.split())>1:
            #Check WordNet part of speech for last word in phrase
            concept_last_word = concept.split()[-1]
            if len(wn.synsets(concept_last_word, pos=wn.VERB)) > 0:
                concept_pos_mat[concept_list.index(concept), pos.index('V')] = 1
                present = True
            if len(wn.synsets(concept_last_word, pos=wn.NOUN)) > 0:
                concept_pos_mat[concept_list.index(concept), pos.index('N')] = 1
                present = True
            if len(wn.synsets(concept_last_word, pos=wn.ADJ)) > 0:
                concept_pos_mat[concept_list.index(concept), pos.index('A')] = 1
                present = True

        if concept in pos_dict:
            pos_str = pos_replace(pos_dict[concept])
            pos_ind = [pos.index(x) for x in pos_str]
            concept_moby_mat[concept_list.index(concept), pos_ind] = 1
            present = True

        #If concept still not identified, check lemma's part of speech
        #This doesn't appear to catch many additional words
        # elif not present:
        #     concept_stems = set([ps.stem(concept)])
        #     concept_stems.add(wnl.lemmatize(concept))
        #
        #     for t in concept_stems:
        #         if t in pos_dict_stem:
        #             pos_str = pos_replace(pos_dict[pos_dict_stem[t]])
        #             pos_ind = [pos.index(x) for x in pos_str]
        #             concept_pos_mat[concept_list.index(concept), pos_ind] = 1
        #             present = True
        #
        #         if len(wn.synsets(t, pos=wn.VERB)) > 0:
        #             concept_pos_mat[concept_list.index(concept), pos.index('V')] = 1
        #             present = True
        #         if len(wn.synsets(t, pos=wn.NOUN)) > 0:
        #             concept_pos_mat[concept_list.index(concept), pos.index('N')] = 1
        #             present = True
        #         if len(wn.synsets(t, pos=wn.ADJ)) > 0:
        #             concept_pos_mat[concept_list.index(concept), pos.index('A')] = 1
        #             present = True

        if not present:
            #print "No part of speech for " + concept
            no_pos.append(concept)

    with open(os.path.join(save_dir, search_descriptor+'_no_pos.txt'), 'w') as f:
        f.write("\n".join(no_pos))

    save_path = os.path.join(save_dir, search_descriptor+'_pos.mat')
    sp.savemat(save_path, {"pos_wordnet": concept_pos_mat, "pos_moby": concept_moby_mat, "concepts": concept_list})

    print("done parse parts of speech")
    print("Number concepts without part of speech: {}".format(len(no_pos)))
    return concept_pos_mat
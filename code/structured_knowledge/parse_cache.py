# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:00:07 2013

Collects all edges from the cashed ConceptNet searches that are also relevent to the 
Flickr data. Stores as pickel file. 

@author: mauceri2
"""
import csv
import json
import pickle
import os.path
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import numpy as np
import scipy.io as spio
import collections

def vet_candidates(data_root, knowledge_base_str, candidate_vocabulary):
    errors = []
    json_path = os.path.join(data_root, knowledge_base_str + "JSON")

    confirmed_concepts = set([])
    concept_freq_dict = {t[0]: t[1] for t in candidate_vocabulary}
    candidate_vocabulary_list = concept_freq_dict.keys()
    # get all the edges
    for concept in candidate_vocabulary_list:
        if concept.replace(" ", "").isalpha():
            if knowledge_base_str == "ConceptNet":
                concept_save_str = concept.replace(" ", "_")
            else:
                concept_save_str = concept
            cache_file_path = os.path.join(json_path, concept_save_str + '.json')
            if (os.path.exists(cache_file_path)):
                with open(cache_file_path, 'r') as c:
                    jsonString = c.read()

                    try:
                        jsonStructure = json.loads(jsonString)

                    except:
                        #print "Error on " + cache_file_path
                        errors.append(concept)

                    else:
                        concept_set = set(concept.split())
                        if knowledge_base_str == "Freebase":
                            if not len(jsonStructure) == 0:
                                source_sense = jsonStructure[0][0].split('/')[-1].replace("_", " ")
                                source_sense_set = set(source_sense.split())
                                if (source_sense_set == concept_set):
                                    confirmed_concepts.add((source_sense, concept_freq_dict[concept]))
                        else:
                            if "numFound" in jsonStructure and jsonStructure["numFound"] > 0:
                                for edge in jsonStructure["edges"]:
                                    source_sense = edge['start'].split('/')[-1].replace("_", " ")
                                    source_sense_set = set(source_sense.split())
                                    if (source_sense_set == concept_set):
                                        confirmed_concepts.add((source_sense, concept_freq_dict[concept]))
                                    break
    print "Vet candidate vocab finished"
    return confirmed_concepts


def remove_spaces_and_stopwords(phrase, stopwords_set):
    return "".join([w for w in phrase.split() if not w in stopwords_set])


def remove_stopwords(phrase, stopwords_set):
    return " ".join([w for w in phrase.split() if not w in stopwords_set])

def remove_spaces(phrase):
    return "".join(phrase.split())

def process_cache(knowledge_dir, knowledge_base_str, freq_vocabulary, do_parse_cache, save_descriptor, stats_dir):
    print "Starting Cache Parse"
    errors = []
    json_path = os.path.join(knowledge_dir, knowledge_base_str + "JSON")
    stopwords_set = set(stopwords.words('english'))

    freq_vocabulary_list = [t[0] for t in freq_vocabulary]
    freq_vocabulary_list_no_spaces = [remove_spaces(t[0]) for t in freq_vocabulary]
    common_noun_set = set([w for w in freq_vocabulary_list_no_spaces if len(wn.synsets(w)) > 0])
    freq_vocabulary_freq = [t[1] for t in freq_vocabulary]

    freq_vocabulary_dict = dict(zip(freq_vocabulary_list + freq_vocabulary_list_no_spaces, range(0, len(freq_vocabulary_list)) + range(0, len(freq_vocabulary_list))))

    freq_vocabulary_set = set(freq_vocabulary_list)
    freq_vocabulary_set_no_spaces = set(freq_vocabulary_list_no_spaces)

    num_vocabulary = len(freq_vocabulary_list)

    candidate_vocabulary = set([])
    candidate_synonyms = set([])
    synonyms_mat = np.zeros((num_vocabulary, num_vocabulary))
    word_sense = set([])
    concepts_in_freebase = set([])
    type_dict = collections.defaultdict(list)
    population_dict = collections.defaultdict(set)

    if do_parse_cache:
        edges = {}
        relations = {}
        all_relations = set([])
        concepts = {freq_vocabulary_dict[concept[0]]: {"tag": concept[0],
                                                       "flickr_score": concept[1],
                                                       "group": 1,
                                                       "edge_URI_source": [],
                                                       "edge_URI_target": [],
                                                       "categories": []} for concept in freq_vocabulary}

    # get all the edges
    for ii in range(0, num_vocabulary):
        concept = freq_vocabulary_list[ii]
        concept_no_spaces = freq_vocabulary_list_no_spaces[ii]

        if concept.replace(" ", "").isalpha():
            concept_save_str = concept
            if knowledge_base_str == "ConceptNet":
                concept_save_str = concept.replace(" ", "_")

            cache_file_path = os.path.join(json_path, concept_save_str + '.json')

            if (os.path.exists(cache_file_path)):
                with open(cache_file_path, 'r') as fp:
                    try:
                        jsonStructure = json.load(fp)

                    except:
                        print "Error on " + cache_file_path
                        errors.append(concept)

                    else:
                        source_sense = concept
                        concept_results = {}
                        aliases = []

                        #Identify correct result from Freebase search
                        if knowledge_base_str == "Freebase":
                            relation_key = "property"

                            if not len(jsonStructure) == 0:
                                source_senses = [sense[0].split('/')[-1].replace("_", " ").lower().encode("ascii", "replace") for sense in jsonStructure]
                                source_senses_no_spaces = [remove_spaces(phrase) for phrase in source_senses]
                                if concept_no_spaces in source_senses_no_spaces:
                                    sense_index = source_senses_no_spaces.index(concept_no_spaces)
                                    concept_results = jsonStructure[sense_index][1]
                                    source_sense = source_senses[sense_index]

                                    if (relation_key in concept_results) and ("/common/topic/alias" in concept_results[relation_key]):
                                        aliases = [values['value'].lower().encode("ascii", "replace") for values in concept_results[relation_key]["/common/topic/alias"]["values"]]
                                else:
                                    source_sense_names = []
                                    source_sense_aliases = []
                                    source_sense_names_no_spaces = []
                                    source_sense_aliases_no_spaces = []

                                    for sense in jsonStructure:
                                        if not "error" in sense[1]:
                                            if "/common/topic/alias" in sense[1]["property"]:
                                                alias_values = [values['value'].lower().encode("ascii", "replace") for values in sense[1]["property"]["/common/topic/alias"]["values"]]
                                                source_sense_aliases.append(alias_values)
                                                source_sense_aliases_no_spaces.append([remove_spaces(phrase) for phrase in alias_values])
                                            else:
                                                source_sense_aliases.append([])
                                                source_sense_aliases_no_spaces.append([])

                                            if "/type/object/name" in sense[1]["property"]:
                                                names = [values['value'].lower().encode("ascii", "replace") for values in sense[1]["property"]["/type/object/name"]["values"]]
                                                source_sense_names.append(names)
                                                source_sense_names_no_spaces.append([remove_spaces(phrase) for phrase in names])
                                            else:
                                                source_sense_names.append([])
                                                source_sense_names_no_spaces.append([])
                                        else:
                                            source_sense_aliases.append([])
                                            source_sense_aliases_no_spaces.append([])
                                            source_sense_names.append([])
                                            source_sense_names_no_spaces.append([])
                                            print("Error in json: {}".format(concept))

                                    name_mask = [concept_no_spaces in names for names in source_sense_names_no_spaces]
                                    if any(name_mask):
                                        sense_index = name_mask.index(True)
                                        concept_results = jsonStructure[sense_index][1]
                                        word_index = source_sense_names_no_spaces[sense_index].index(concept_no_spaces)
                                        source_sense = source_sense_names[sense_index][word_index]

                                        aliases = source_sense_aliases[sense_index]
                                    else:
                                        alias_mask = [concept_no_spaces in aliases for aliases in source_sense_aliases_no_spaces]
                                        if any(alias_mask):
                                            sense_index = alias_mask.index(True)
                                            concept_results = jsonStructure[sense_index][1]
                                            word_index = source_sense_aliases_no_spaces[sense_index].index(concept_no_spaces)
                                            source_sense = source_sense_aliases[sense_index][word_index]

                                            aliases = source_sense_aliases[sense_index]

                            # if not concept_results == {}:
                            #     print source_sense

                        else:
                            concept_results = jsonStructure
                            relation_key = "edges"

                        if relation_key in concept_results:
                            for edge in concept_results[relation_key]:
                                start = concept
                                endList = []
                                if knowledge_base_str == "Freebase":
                                    relation = edge
                                    currentRelation = concept_results[relation_key][relation]
                                    delimitRelation = relation.split('/')
                                    if (not delimitRelation[1] == 'common' and
                                            not delimitRelation[1] == "type"):
                                        endList = [v["text"].lower() for v in currentRelation["values"]]

                                    if relation == "/common/topic/notable_for" or relation == "/type/object/type":
                                        typeList = [v["id"] for v in currentRelation["values"]]

                                        if relation == "/type/object/type":
                                            typeList = [t for t in typeList if t == "/location/location"]

                                        type_dict[start].extend(typeList)

                                    if relation == "/location/statistical_region/population":
                                        population_text = [v["text"].split(' - ')[0:2] for v in currentRelation["values"]]
                                        population_text.sort(key= lambda x: x[1])
                                        population_dict[start].add(population_text[-1][0])

                                else:
                                    delimitRelation = edge['rel'].split('/')
                                    relation = delimitRelation[-1]
                                    start = edge['startLemmas'].lower().encode("ascii", "replace")
                                    # TODO Need more sophisticated parsing for start field
                                    # if len(edge['start'].split('c/en')) > 1:
                                    #     source_sense = edge['start'].split('c/en/')[1].split('/')[0].replace("_", " ")
                                    # else:
                                    source_sense = start
                                    endList.append(edge['endLemmas'])

                                for alias in aliases:
                                    #Remove spaces
                                    alias_no_space = remove_spaces(alias)

                                    #Check for vocabulary match
                                    #Concatenated phrase is in vocabulary, replace with phrase
                                    #And concatenated phrase is not common noun
                                    if ((alias_no_space in freq_vocabulary_set_no_spaces) and (not alias in freq_vocabulary_set)
                                        and (not alias_no_space in common_noun_set)):
                                        if len(alias) > len(freq_vocabulary_list[freq_vocabulary_dict[alias_no_space]]):
                                            freq_vocabulary_list[freq_vocabulary_dict[alias_no_space]] = alias
                                            freq_vocabulary_dict[alias] = freq_vocabulary_dict[alias_no_space]
                                            freq_vocabulary_set.add(alias)

                                    #All words in phrase are in vocabulary
                                    alias_set = set(alias.split())
                                    if alias_set.issubset(freq_vocabulary_set) and (not alias in freq_vocabulary_set):
                                            candidate_vocabulary.add(alias)

                                if source_sense in freq_vocabulary_set:
                                    start = source_sense
                                    concepts_in_freebase.add(start)
                                else:
                                    source_sense_no_spaces = remove_spaces(source_sense)
                                    if source_sense_no_spaces in freq_vocabulary_set and not source_sense_no_spaces in common_noun_set:
                                        source_sense_no_spaces = remove_spaces(source_sense)
                                        start = source_sense
                                        concepts_in_freebase.add(start)
                                        freq_vocabulary_list[freq_vocabulary_dict[source_sense_no_spaces]] = source_sense
                                        freq_vocabulary_dict[source_sense] = freq_vocabulary_dict[source_sense_no_spaces]
                                        freq_vocabulary_set.add(source_sense)
                                    else:
                                        word_sense.add((start, source_sense))

                                #If alias in vocabulary, add to synonyms list
                                for alias in aliases:
                                    if (not alias == start) and alias in freq_vocabulary_set:
                                        candidate_synonyms.add((start, alias))
                                        synonyms_mat[freq_vocabulary_dict[start], freq_vocabulary_dict[alias]] = 1
                                        synonyms_mat[freq_vocabulary_dict[alias], freq_vocabulary_dict[start]] = 1

                                for end in endList:
                                    #Check for vocabulary match
                                    #Concatenated phrase is in vocabulary and not in common noun set, replace with phrase
                                    end_no_spaces = remove_spaces(end)
                                    if ((end_no_spaces in freq_vocabulary_set_no_spaces) and (not end in freq_vocabulary_set)
                                        and (not end_no_spaces in common_noun_set)):
                                        freq_vocabulary_list[freq_vocabulary_dict[end_no_spaces]] = end
                                        freq_vocabulary_list_no_spaces[freq_vocabulary_dict[end_no_spaces]] = end_no_spaces
                                        freq_vocabulary_freq[freq_vocabulary_dict[end_no_spaces]] = freq_vocabulary[freq_vocabulary_dict[end_no_spaces]][1]

                                        freq_vocabulary_dict[end] = freq_vocabulary_dict[end_no_spaces]
                                        freq_vocabulary_set.add(end)
                                        freq_vocabulary_set_no_spaces.add(end_no_spaces)

                                    #All words in phrase are in vocabulary
                                    end_set = set(end.split())
                                    if end_set.issubset(freq_vocabulary_set):
                                        if not end in freq_vocabulary_set:
                                            candidate_vocabulary.add(end)

                                    if do_parse_cache and (end in freq_vocabulary_set):
                                        if (start in freq_vocabulary_set) and (not end == start):
                                            #Add edge if new
                                            if knowledge_base_str == "Freebase":
                                                uri = start + relation + '/' + '_'.join(end.split())
                                                weight = 1
                                            else:
                                                uri = edge["uri"]
                                                weight = edge["weight"]

                                            if not (uri in edges):
                                                edges[uri] = {"source": freq_vocabulary_dict[start], "source_sense": source_sense,
                                                              "target": freq_vocabulary_dict[end], "relation": relation, "uri": uri,
                                                              "weight": weight}

                                            concepts[freq_vocabulary_dict[start]]['edge_URI_source'].append(uri)
                                            concepts[freq_vocabulary_dict[end]]['edge_URI_target'].append(uri)

                                            #Add relation if new
                                            all_relations.add(relation)
                                            if not relation in relations:
                                                relations[relation] = {"color": "", "visible": True}

        else:
            errors.append(concept)
            #print "Error on " + concept + "\n"

    knowledge_save_dir = os.path.join(knowledge_dir, knowledge_base_str)
    if not os.path.exists(knowledge_save_dir):
        os.mkdir(knowledge_save_dir)

    if do_parse_cache:
        #Save the structured graph
        print "Saving Cache Parse"
        save_path = os.path.join(knowledge_save_dir, '{}_{}.pkl'.format(save_descriptor, knowledge_base_str))
        if not os.path.exists(knowledge_save_dir):
            os.makedirs(knowledge_save_dir)

        with open(save_path, 'wb') as save:
            pickle.dump({'tags': concepts, 'edges': edges, 'relations': relations, 'vocabulary': freq_vocabulary_list}, save)

        #Save list of relationships
        save_path = os.path.join(knowledge_save_dir, '{}_{}_relations.txt'.format(save_descriptor, knowledge_base_str))
        all_relations = list(all_relations)
        all_relations.sort()
        with open(save_path, 'wb') as save:
            save.write("\n".join(all_relations))

        if knowledge_base_str == "Freebase":
            struct_save_dir = stats_dir

            #Save synonyms
            candidate_synonyms = list(candidate_synonyms)
            save_path = os.path.join(struct_save_dir, save_descriptor + '_synonyms.txt')
            with open(save_path, 'wb') as save:
                save.write("\n".join(["\t".join(t) for t in candidate_synonyms]).encode('ascii', 'replace'))

            save_path = os.path.join(struct_save_dir, save_descriptor + '_aliases.mat')
            spio.savemat(save_path, {'aliases_mask': synonyms_mat, 'vocabulary':freq_vocabulary_list})

            #Save types
            all_types = list(set([v for v_list in type_dict.values() for v in v_list if len(v)>0]))
            num_types = len(all_types)
            all_types_dict = dict(zip(all_types, range(0, num_types)))

            super_types = list(set([v.split('/')[1] for v in all_types]))
            num_super_types = len(super_types)
            super_types_dict = dict(zip(super_types, range(0, num_super_types)))

            types_mat = np.zeros((num_vocabulary, num_types))
            super_types_mat = np.zeros((num_vocabulary, num_super_types))
            for concept in type_dict:
                concept_ind = freq_vocabulary_dict[concept]
                type_ind = [all_types_dict[v] for v in type_dict[concept] if len(v)>0]
                super_type_ind = [super_types_dict[v.split('/')[1]] for v in type_dict[concept] if len(v)>0]
                types_mat[concept_ind, type_ind] = 1
                super_types_mat[concept_ind, super_type_ind] = 1
            save_path = os.path.join(struct_save_dir, save_descriptor + '_types.mat')
            spio.savemat(save_path, {'types_mask': types_mat, 'types_str': all_types, 'vocabulary':freq_vocabulary_list})

            save_path = os.path.join(struct_save_dir, save_descriptor + '_super_types.mat')
            spio.savemat(save_path, {'super_types_mask': super_types_mat, 'super_types_str': super_types, 'vocabulary':freq_vocabulary_list})

            #Save Location mask
            location_ind = super_types.index('location')
            location_mat = super_types_mat[:, location_ind]
            save_path = os.path.join(struct_save_dir, save_descriptor + '_locations.mat')
            spio.savemat(save_path, {'location_mask': location_mat, 'vocabulary':freq_vocabulary_list})

            population_stat, indices = zip(*[(max(list(population_dict[location])), freq_vocabulary_dict[location])
                            for location in population_dict.keys()])
            population_vec = np.zeros((num_vocabulary,))
            population_vec[list(indices)] = population_stat
            save_path = os.path.join(struct_save_dir, save_descriptor + '_population.mat')
            spio.savemat(save_path, {'population_mask': population_vec, 'vocabulary':freq_vocabulary_list})

    #Save list of errors
    save_path = os.path.join(knowledge_save_dir, save_descriptor + '_errors.txt')
    with open(save_path, 'wb') as save:
        save.write("\n".join(errors))

    #Save newly discovered tags
    candidate_vocabulary = list(candidate_vocabulary)
    save_path = os.path.join(knowledge_save_dir, "candidate_concepts", save_descriptor + '_candidates.txt')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'wb') as save:
        save.write("\n".join(candidate_vocabulary).encode('ascii', 'replace'))

    #Vocabulary with and without spaces
    freq_vocabulary = zip(freq_vocabulary_list, freq_vocabulary_freq, freq_vocabulary_list_no_spaces)

    errors = list(set(errors))
    print "Done with cache parse"
    return freq_vocabulary, candidate_vocabulary, errors


def find_candidates(knowledge_dir, knowledge_base_str, freq_vocabulary, save_descriptor):
    return process_cache(knowledge_dir, knowledge_base_str, freq_vocabulary, False, save_descriptor, "")


def parse_cache(knowledge_dir, knowledge_base_str, freq_vocabulary, save_descriptor, stats_dir):
    return process_cache(knowledge_dir, knowledge_base_str, freq_vocabulary, True, save_descriptor, stats_dir)


if __name__ == '__main__':
    search = "cat"
    num_tags = 100
    type = 'tags'
    source_knowledge_base = "Freebase"
    search_descriptor = "{}_{}_{}".format(search, num_tags, type)

    tag_root = "E:\\data\\Iconic\data\\test_crawler\\data\\concepts"
    data_root = "E:\data\StructuredKnowledge"
    tag_path = os.path.join(tag_root, search_descriptor + '_all_concepts.txt')
    parse_cache(search_descriptor, source_knowledge_base, tag_path, data_root, False)

                
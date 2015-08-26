# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:45:53 2014

@author: mauceri2

Creates a dictionary where each entry (source, target, relationship) represents the number of ConceptNet edges with those variables.
I didn't use a matrix because the data is very sparse and the size of the matrix would be prohibitive to storing it in a Matlab readable format.
As is, the dictionary is stored as a json file, which is then read and converted by the matlab script parse_json_matlab.m

The output is 
{'netmatrix':{...}, 'conceptScoreMatrix':{...}, 'dims':{'dims':(...), 'tags':{...}, 'relations':{...}}
Where 
    netmatrix is the dictionary of edge occurances
    conceptScoreMatrix is a similar dictionary with the average edge scores for each (source, target, relationship) 
    dims are the dimentions (tags, tags, relationships)
    tags is a list of tags
    relations is a list of relationships
"""

import pickle
import ujson
import os
import numpy as np
import scipy.io as sp
import csv
from scipy.sparse import lil_matrix

def sub2ind(x, y, shape):
    return x*shape[1] + y

def build_adjacency_matrices(knowledge_dir, save_dir, concept_list, search_descriptor):
    num_vocabulary = len(concept_list)
    sources = ["ConceptNet", "Freebase"]
    errors = []

    for source in sources:

        pkl_path = os.path.join(knowledge_dir, source, search_descriptor + "_" + source + ".pkl")
        with open(pkl_path) as f:
            net = pickle.load(f)

        relations = net['relations'].keys()
        num_relations = len(relations)
        relations_dict = dict(zip(relations, range(0, num_relations)))

        adjacency_matrix_sparse = lil_matrix((num_vocabulary * num_vocabulary, num_relations))
        weighted_adjacency_sparse = lil_matrix((num_vocabulary * num_vocabulary, num_relations))

        for edge in net['edges']:
            edge = net['edges'][edge]
            concept_indices = [edge['source'], edge['target']]

            relation = edge['relation']
            relation_index = relations_dict[relation]

            adjacency_matrix_sparse[sub2ind(edge['source'],  edge['target'], [num_vocabulary, num_vocabulary]), relation_index] += 1
            weighted_adjacency_sparse[sub2ind(edge['source'],  edge['target'], [num_vocabulary, num_vocabulary]), relation_index] += edge['weight']

        path = os.path.join(save_dir, '{}_{}_adjacency.mat'.format(search_descriptor, source))
        dims = {'dims': ('source', 'target', 'relation'), 'tags': concept_list, 'relations': relations}

        sp.savemat(path, {'adjacency': adjacency_matrix_sparse, 'weighted_adjacency': weighted_adjacency_sparse, 'attributes': dims, 'shape': [num_vocabulary, num_vocabulary, num_relations]})

if __name__ == '__main__':
    import sys
    if len(sys.argv)<6:
        print 'Too few arguments. Execute as >> python build_adjacency_matrices knowledge_dir root_dir query conceptKeyList numConcepts'
        #Example: E:\data\StructuredKnowledge E:\data\Iconic\data\ cat tags 6000

    sys.path.append("C:\Users\mauceri2\Documents\SVN_trunk\Iconic\\flickr\database_builder")
    from get_photo_meta import get_concept_frequency
    from tools.query_descriptor import query_descriptor as query_descriptor

    concept_path = os.path.join(sys.argv[2], 'test_crawler', 'data', 'concepts')
    concept_list = get_concept_frequency(concept_path, sys.argv[3], int(sys.argv[5]), [sys.argv[4]], 'all_concepts')
    search_descriptor = query_descriptor(sys.argv[3], int(sys.argv[5]), [sys.argv[4]])

    save_dir = os.path.join(sys.argv[2], "structure")
    build_adjacency_matrices(sys.argv[1], save_dir, concept_list, search_descriptor)
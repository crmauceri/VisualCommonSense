__author__ = 'mauceri2'
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
import numpy as np
import scipy.io as sp
import os
import json
import csv

def parse_scene_concepts(knowledge_root, save_dir, concept_list, search_descriptor):
    print "begin SUN scene labeling"
    scenes = set([])
    sun_root = os.path.join(knowledge_root, 'SunScene')
    with open(os.path.join(sun_root, 'hierarchy_three_levels', 'three_levels_SUN908.csv'), 'r') as f:
        for line in f:
            if line[1] == '/':
                fields = line.split(',')
                scene_list = fields[0][1:-1].split('/')[2:]
                scene_list = [x.replace('_', ' ') for x in scene_list]
                sub_scenes = [' '.join(scene_list[:x]) for x in range(1, len(scene_list) + 1)]
                scenes = scenes.union(set(sub_scenes))

    scenes = sorted(scenes)
    #print '\n'.join(scenes)

    scene_concept = [x for x in scenes if x in concept_list]
    with open(os.path.join(save_dir, search_descriptor+'_SUN.txt'), 'w') as f:
        f.write('\n'.join(scene_concept))

    scene_concept_feature = map(lambda x: 1 if x in scenes else 0, concept_list)
    sp.savemat(os.path.join(save_dir, search_descriptor+'_SUN_feature.mat'), {'SUN': np.array(scene_concept_feature), 'concepts': concept_list})

    with open(os.path.join(sun_root, 'three_levels_SUN908.txt'), 'w') as f:
        f.write('\n'.join(scenes))

    print "done with SUN scene labeling"

def explore_tree(cur_node, parent_ind, concept_list, object_mat, nouns):
    if 'words' in cur_node.attrib:
        cur_concepts = cur_node.attrib['words'].lower().split(',')
        valid_concepts = [concept_list.index(x.strip()) for x in cur_concepts if x.strip() in concept_list]
        if len(valid_concepts) > 0:
            if parent_ind > -1:
                for concept in valid_concepts:
                    object_mat[parent_ind, concept] = 1
            parent_ind = valid_concepts
        nouns += [x.strip() for x in cur_concepts if x.strip() in concept_list]

    for child in cur_node.findall('synset'):
        object_mat, nouns = explore_tree(child, parent_ind, concept_list, object_mat, nouns)

    return object_mat, nouns

def parse_object_concepts(knowledge_dir, save_dir, concept_list, search_descriptor):
    print "begin ImageNet labeling"
    num_concepts = len(concept_list)
    struct_tree = ET.parse(os.path.join(knowledge_dir, 'ImageNet', 'structure_released.xml'))
    struct_root = struct_tree.getroot()

    # meta_tree = ET.parse(os.path.join(data_root, 'ReleaseStatus.xml'))
    # meta_root = meta_tree.getroot()
    #
    # concepts = set([])
    # for node in meta_root.iter('synset'):
    #     node_attrib = node.attrib
    #     conceptList = [x.strip() for x in node_attrib['words'].lower().split(',')]
    #     concepts = concepts.union(set(conceptList))

    #Is object representation
    object_concepts = []
    for node in struct_root.iter('synset'):
        node_attrib = node.attrib
        object_concepts += [x.strip() for x in node_attrib['words'].lower().split(',') if x.strip() in concept_list]

    object_concept_feature = map(lambda x: 1 if x in object_concepts else 0, concept_list)
    sp.savemat(os.path.join(save_dir, search_descriptor+'_isImageNet_feature.mat'), {'isImageNet': np.array(object_concept_feature), 'concepts': concept_list})

    object_concepts = list(set(object_concepts))
    with open(os.path.join(save_dir, search_descriptor+'_imageNet.txt'), 'w') as f:
        f.write('\n'.join(sorted(object_concepts)))

    #Graph representation
    object_mat = np.zeros((num_concepts, num_concepts))
    cur_node = struct_root
    object_mat, nouns = explore_tree(cur_node, -1, concept_list, object_mat, [])

    #Debugging check for whether all nodes were visited
    #print [x for x in object_concepts if not x in nouns]

    sp.savemat(os.path.join(save_dir, search_descriptor+'_imageNet_feature.mat'), {'imagenet': object_mat, 'tags': concept_list})
    print "done with ImageNet labels"

if __name__ == '__main__':
    import sys
    if len(sys.argv)<6:
        print 'Too few arguments. Execute as >> python parse_object_scene.py knowledge_dir root query conceptKeyList numConcepts'
        #Example: E:\data\StructuredKnowledge E:\data\Iconic\data cat tags 6000

    sys.path.append("C:\Users\mauceri2\Documents\SVN_trunk\Iconic\\flickr\database_builder")
    from get_photo_meta import get_concept_list
    from tools.query_descriptor import query_descriptor as query_descriptor

    concept_path = os.path.join(sys.argv[2], 'test_crawler', 'data', 'concepts')
    concept_list = get_concept_list(concept_path, sys.argv[3], int(sys.argv[5]), [sys.argv[4]], 'by_owners')
    search_descriptor = query_descriptor(sys.argv[3], int(sys.argv[5]), [sys.argv[4]])

    save_dir = os.path.join(sys.argv[2], "structure")
    parse_object_concepts(sys.argv[1], save_dir, concept_list, search_descriptor)
    parse_scene_concepts(sys.argv[1], save_dir, concept_list, search_descriptor)

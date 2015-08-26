root = 'E:\data\Iconic\data'; % 'E:\rooms\data\'; % 
glove_name = 'glove.42B.300d'; %'glove.twitter.27B.200d', 'glove.6B.300d', 'vectors.840B.300d'
search_description = 'combined_animal_rooms'; %'rooms_200_extended'; % 'cat', 'chair', 'kitchen', 'shoe' 'biology_domesticated_animal' 'fashion_garment'

use_threshold = false;
use_merged_relations = true; 
use_feature_difference = true; 
min_num_images = 100;
min_images_unlabeled = 100;
min_num_edges = 50;
use_approx = false;

rooms_root = 'E:/rooms/data';
search_description_rooms = 'rooms_200_extended';
feature_dir = sprintf('%s\\word2vec_features\\', rooms_root);
rooms_feature_path = [feature_dir search_description_rooms '_feature_' glove_name '.mat'];
rooms_word2vec = load(rooms_feature_path);
rooms_vocab = cellstr(rooms_word2vec.concepts);

animals_root = 'E:\data\Iconic\data';
search_description_animals = 'biology_domesticated_animal_500_extended';
feature_dir = sprintf('%s\\word2vec_features\\', animals_root);
animals_feature_path = [feature_dir search_description_animals '_feature_' glove_name '.mat'];
animals_word2vec = load(animals_feature_path);
animals_vocab = cellstr(animals_word2vec.concepts);

save_str = 'crosstraining_training_animals';
[cubic_result, gauss_result, error_rates, edges_per_relation] = run_crosstraining(root, ...
    search_description, 0, use_feature_difference, ...
    1, glove_name, use_threshold, use_merged_relations, min_num_images, ...
    min_num_edges, use_approx, save_str, animals_vocab, rooms_vocab, animals_root, rooms_root, ...
    search_description_animals, search_description_rooms);
header = {'Classifier', 'Accuracy', 'Recall' , 'Precision', 'Recall@3', 'Precision@3'};
error_rates_result = [header; [{'Cubic'; 'Gauss'}, num2cell(error_rates)]];
save_path = sprintf('%s/transfer/%s_%s_multi_class_error.txt', root, search_description, save_str);
cell2csv(save_path, error_rates_result, '\t');

combined_save = sprintf('%s/transfer/%s_%s_result.txt', root, search_description, save_str);
cell2csv(combined_save, cubic_result, '\t');

save_str = 'crosstraining_training_rooms';
[cubic_result, gauss_result, error_rates, edges_per_relation] = run_crosstraining(root, ...
    search_description, 0, use_feature_difference, ...
    1, glove_name, use_threshold, use_merged_relations, min_num_images, ...
    min_num_edges, use_approx, save_str, rooms_vocab, animals_vocab, rooms_root, animals_root, ...
    search_description_rooms, search_description_animals);
header = {'Classifier', 'Accuracy', 'Recall' , 'Precision', 'Recall@3', 'Precision@3'};
error_rates_result = [header; [{'Cubic'; 'Gauss'}, num2cell(error_rates)]];
save_path = sprintf('%s/transfer/%s_%s_multi_class_error.txt', root, search_description, save_str);
cell2csv(save_path, error_rates_result, '\t');

combined_save = sprintf('%s/transfer/%s_%s_result.txt', root, search_description, save_str);
cell2csv(combined_save, cubic_result, '\t');
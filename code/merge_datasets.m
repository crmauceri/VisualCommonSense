%INITIALIZE
root = 'E:\data\Iconic\data';
search_description = 'combined_animal_rooms';
rooms_root = 'E:/rooms/data';
animals_root = 'E:\data\Iconic\data';
search_description_animals = 'biology_domesticated_animal_500_extended';
search_description_rooms = 'rooms_200_extended';

%Singular
%FEATURES
glove_name = 'glove.42B.300d';
feature_dir = sprintf('%s\\word2vec_features\\', rooms_root);
rooms_feature_path = [feature_dir search_description_rooms '_feature_' glove_name '.mat'];
feature_dir = sprintf('%s\\word2vec_features\\', animals_root);
animals_feature_path = [feature_dir search_description_animals '_feature_' glove_name '.mat'];

rooms_word2vec = load(rooms_feature_path);
rooms_word2vec.concepts = cellstr(rooms_word2vec.concepts);

animals_word2vec = load(animals_feature_path);
animals_word2vec.concepts = cellstr(animals_word2vec.concepts);

rooms_only = ~ismember(rooms_word2vec.concepts, animals_word2vec.concepts);

concepts = [animals_word2vec.concepts; rooms_word2vec.concepts(rooms_only)];
feature_mask = [animals_word2vec.feature_mask, rooms_word2vec.feature_mask(rooms_only)];
features = [animals_word2vec.features; rooms_word2vec.features(rooms_only, :)];
mean_feature_mask = [animals_word2vec.mean_feature_mask, rooms_word2vec.mean_feature_mask(rooms_only)];

feature_dir = sprintf('%s\\word2vec_features\\', root);
feature_path = [feature_dir search_description '_feature_' glove_name '.mat'];
save(feature_path, 'concepts', 'features', 'feature_mask', 'mean_feature_mask');

%DATA
animals_data = load(sprintf('%s/Flickr_concepts/%s_analysis.mat', animals_root, search_description_animals));
rooms_data = load(sprintf('%s/Flickr_concepts/%s_analysis.mat', rooms_root, search_description_rooms));
header = animals_data.header;
language_iso = animals_data.language_iso;
data = [animals_data.data; rooms_data.data(rooms_only, :)];
save(sprintf('%s/Flickr_concepts/%s_analysis.mat', root, search_description), 'data', 'header', 'concepts', 'language_iso');

%Pairs
% Find the indices that correspond to elements of A
[animal_overlap, ~] = ismember(concepts, animals_word2vec.concepts); 
animal_overlap = double(animal_overlap);
animal_mask = (animal_overlap*animal_overlap')==1;

% Find the indices that correspond to elements of B
[rooms_overlap, rooms_order] = ismember(concepts, rooms_word2vec.concepts); 
rooms_overlap = double(rooms_overlap);
rooms_mask = (rooms_overlap*rooms_overlap')==1;
[x, y] = find(rooms_mask);
rooms_ind = sub2ind([length(rooms_word2vec.concepts), length(rooms_word2vec.concepts)], rooms_order(x), rooms_order(y));

% CODATA
animal_codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', animals_root, search_description_animals));
rooms_codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', rooms_root, search_description_rooms));
new_size = length(concepts)^2; 
data = zeros(new_size, 14); 
data(animal_mask(:), :) = animal_codata.data;
data(rooms_mask(:), :) = max(data(rooms_mask(:), :), rooms_codata.data(rooms_ind, :));
dataDims = [length(concepts), length(concepts), 14];
save(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', root, search_description), 'data', 'dataDims', '-v7.3');

%PHRASES
animal_phrase = load(sprintf('%s/structure/%s_phrase_mask.mat', animals_root, search_description_animals));
rooms_phrase = load(sprintf('%s/structure/%s_phrase_mask.mat', rooms_root, search_description_rooms));
phrase_mask = zeros(new_size, 1);
phrase_mask(animal_mask(:)) = animal_phrase.phrase_mask(:);
phrase_mask(rooms_mask(:)) = max(phrase_mask(rooms_mask(:)), rooms_phrase.phrase_mask(rooms_ind));
phrase_mask = reshape(phrase_mask, dataDims(1:2));
save(sprintf('%s/structure/%s_phrase_mask.mat', root, search_description), 'phrase_mask');

%RELATIONSHIPS
relations_path = sprintf('%s/structure/%s_manual_merged_adjacency_pos.mat', animals_root, search_description_animals);
animals_relations = load(relations_path);
relations_path = sprintf('%s/structure/%s_manual_merged_adjacency_pos.mat', rooms_root, search_description_rooms);
rooms_relations = load(relations_path);

relation_str = animals_relations.relation_str;
relations = zeros(new_size, length(relation_str));
relations(animal_mask(:), :) = animals_relations.relations;
relations(rooms_mask(:), :) = max(relations(rooms_mask(:), :), rooms_relations.relations(rooms_ind, :));

relations_path = sprintf('%s/structure/%s_manual_merged_adjacency_pos.mat', root, search_description);
save(relations_path, 'relations' ,'relation_str', '-v7.3');

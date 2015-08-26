function [has_relations_mask, in_flickr, sorted_relations_str, relations, ...
    geographic_containment, at_location_geographic, geographic_adjective, ...
    word2vec, edges_per_relation] = initialize_workspace_retrieve(root, search_description, use_glove, ...
    glove_name, use_merged_relations, min_num_images, min_num_edges, pmi_threshold, use_approx)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
variable_def

%Load features
if use_glove
    feature_dir = sprintf('%s\\word2vec_features\\', root);
    feature_path = [feature_dir search_description '_feature_' glove_name '.mat'];
else
    feature_dir = sprintf('%s\\word2vec_features\\', root);
    feature_path = [feature_dir search_description '_feature.mat'];
end
word2vec = load(feature_path);
word2vec.concepts = cellstr(word2vec.concepts);

%Some vocabulary does not have features. Filter this vocabulary
feature_mask = word2vec.feature_mask'; 
feature_mask = feature_mask * feature_mask';

% mean_feature_mask = word2vec.mean_feature_mask'; 
% mean_feature_mask = mean_feature_mask * ones(1, length(word2vec.concepts));
% mean_feature_mask = mean_feature_mask | mean_feature_mask';

%Flickr data
if use_approx
    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_images.mat', root, search_description));
    image_threshold_key = NUM_IMAGES;
else
    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', root, search_description));
    image_threshold_key = NUM_OWNERS;
end
%Do everything requireing codata first to avoid memory issues
dataDims = codata.dataDims;
is_lemma = codata.data(:, IS_LEMMA)==1;

%Ignore relationships between aliass
is_alias = codata.data(:, IS_ALIAS)==1;

%Ignore known translations
is_translation = reshape(codata.data(:, IS_TRANSLATION), dataDims(1:2)) ==1;
is_translation(eye(dataDims(1))==1) = 0;

%Meets threshold values
in_flickr = codata.data(:,image_threshold_key)>=min_num_images;
in_flickr = in_flickr & codata.data(:,N_PMI)>=pmi_threshold;

clear 'codata'


data = load(sprintf('%s/Flickr_concepts/%s_analysis.mat', root, search_description));

%Filter some of the vocabulary
load(sprintf('%s/filter_lists/digital_camera_vocab.mat', root));
camera_vocab = ismember(data.concepts, lower(digital_camera_vocab)); %Camera related words
ambiguous_vocab = ~cellfun('isempty', regexp(data.concepts, '^[?]+$')); %Match all strings containing only question marks;
numbers = ~cellfun('isempty', regexp(data.concepts, '\d+')); %Match any numbers
auto_tags = ~cellfun('isempty', regexp(data.concepts, ':')); %Automatic tags from smart phone apps and flickr
vision_tags = ~cellfun('isempty', regexp(data.concepts, 'vision')); %Flickr's vision system
geography_tags = data.data(:, ismember(data.header, 'isLocation')); %Geographic locations

%Use regex to remove a bit more camera vocabulary
regex_camera_mask = ~cellfun('isempty', regexp(data.concepts, 'photo'));

%Ignore words that do not appear in english and are not locations
language_iso = ismember(data.header, data.language_iso);
is_foreign = any(data.data(:, language_iso), 2) & ~data.data(:, ismember(data.header, 'eng')) ...
    & ~data.data(:, ismember(data.header, 'isProperNoun')) & ~data.data(:, ismember(data.header, 'isCommonNoun'));

%vocab_mask = camera_vocab | ambiguous_vocab | (auto_tags & ~vision_tags) ...
    %| numbers | geography_tags | regex_camera_mask| is_foreign;
vocab_mask = camera_vocab | ambiguous_vocab | (auto_tags & ~vision_tags) ...
    | numbers | regex_camera_mask | is_foreign;
vocab_pair_mask = repmat(vocab_mask, 1, dataDims(1)) | repmat(vocab_mask, 1, dataDims(1))';

%Ignore plurals
is_lemma = reshape(is_lemma, dataDims(1:2));
[plurals, singular] = find(is_lemma);
is_plural = false(dataDims(1:2));
is_plural(plurals, :) = true;
is_plural(:, plurals) = true;

%Ignore relationships between phrases and component words
load(sprintf('%s/structure/%s_phrase_mask.mat', root, search_description));
is_phrase = phrase_mask == 1;

%Ignore relationships between adjectives 
is_adj = data.data(:, ismember(data.header, 'pos_A')) & ~any(data.data(:, ismember(data.header, {'pos_N', 'pos_V'})), 2);
adj_pair_mask = (double(is_adj) * double(is_adj)')>0;

%Ignore relationships between verbs
is_verb = data.data(:, ismember(data.header, 'pos_V')) & ~any(data.data(:, ismember(data.header, {'pos_N', 'pos_A'})), 2);
verb_pair_mask = (double(is_verb) * double(is_verb)')>0;

in_flickr = in_flickr & feature_mask(:) & ~is_plural(:) &~is_alias(:) & ~vocab_pair_mask(:)  ...
    & ~adj_pair_mask(:) & ~verb_pair_mask(:) & ~is_translation(:) & ~phrase_mask(:); % ...
    %& ~mean_feature_mask(:);

%Relations
if use_merged_relations
    relations_path = sprintf('%s/structure/%s_manual_merged_adjacency_pos.mat', root, search_description);
    load(relations_path);
else
    freebase_path = sprintf('%s/structure/%s_Freebase_adjacency_merged.mat', root, search_description);
    freebase = load(freebase_path);
    conceptnet_path = sprintf('%s/structure/%s_ConceptNet_adjacency_merged.mat', root, search_description);
    conceptnet = load(conceptnet_path);
    relations = [conceptnet.adjacent.adjacency freebase.adjacent.adjacency];
    relation_str = [conceptnet.adjacent.attributes.relations; freebase.adjacent.attributes.relations];
    clear freebase conceptnet
end

relations(~in_flickr, :) = 0;
relations_freq = sum(relations, 1);
[edges_per_relation, index] = sort(relations_freq, 'descend');
sorted_relations_str = relation_str(index(edges_per_relation>min_num_edges));
relations = relations(:, index(edges_per_relation>min_num_edges));
has_relations_mask = spones(sum(relations, 2));
num_relationships = size(sorted_relations_str, 1);

% %Relationship coverage
% relation_coverage = reshape(sum(relations, 2), dataDims(1:2));
% relation_coverage = relation_coverage + relation_coverage';
% relation_coverage(eye(size(relation_coverage))==1) = 0;
% edges_per_concept_cell = [data.concepts, num2cell(sum(relation_coverage, 2))];
% edges_per_concept = sum(relation_coverage, 2);
% display(sprintf('Percentage covered: %f', nnz(edges_per_concept)/length(edges_per_concept)));
% 
% %Relationship/Concept overlap
% num_concepts = length(data.concepts);
% concept1_relationship = zeros(num_concepts, num_relationships);
% concept2_relationship = zeros(num_relationships, num_concepts);
% for ii = 1:num_relationships
%    concept1_relationship(:, ii) = sum(reshape(relations(:, ii), [num_concepts, num_concepts]), 1);
%    concept2_relationship(ii, :) = sum(reshape(relations(:, ii), [num_concepts, num_concepts]), 2);
% end
% 
% concept1_overlap = concept1_relationship' * concept1_relationship;
% concept2_overlap = concept2_relationship * concept2_relationship';
% 
% display_confusion(concept1_overlap ./ repmat(sum(concept1_overlap, 1), num_relationships, 1), sorted_relations_str);
% xlabel('Target Relationship');
% ylabel('Relationships Sharing Concept');
% title('Percentage of First Concept Shared');
% 
% display_confusion(concept2_overlap ./ repmat(sum(concept2_overlap, 1), num_relationships, 1), sorted_relations_str);
% xlabel('Target Relationship');
% ylabel('Relationships Sharing Concept');
% title('Percentage of Second Concept Shared');

% Geographic Filter Classifiers
geographic_containment = (double(geography_tags) * double(geography_tags'));
at_location_geographic = (double(geography_tags) * double(data.data(:, ismember(data.header, 'isCommonNoun')))');
geographic_adjective = (double(geography_tags) * double(data.data(:, ismember(data.header, 'pos_A'))'));

% display('Type "return" to close figures and continue');
% keyboard;
% close all
end


function [cubic_result, gauss_result, error_rates] = run_retrieve_multiclass_error(root, ...
    search_description, use_equal_weight, use_diff, use_glove, glove_name, use_threshold, ...
    use_merged_relations, min_num_images, min_num_edges, use_approx, save_str)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
addpath('transfer');
addpath('analysis');
addpath('prec_rec');
variable_def;

[has_relations_mask, in_flickr, sorted_relations_str, relations, ...
    geographic_containment, at_location_geographic, geographic_adjective, ...
    word2vec, edges_per_relation] = initialize_workspace_retrieve(root, search_description, use_glove, glove_name, ...
    use_merged_relations, min_num_images, min_num_edges, -1, use_approx);

%Flickr data
if use_approx
    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_images.mat', root, search_description));
else
    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', root, search_description));
end
data = load(sprintf('%s/Flickr_concepts/%s_analysis.mat', root, search_description));

%The edges we want to label for cross fold validation error
all_edges = find(has_relations_mask & in_flickr);
number_of_images = codata.data(all_edges, NUM_IMAGES);
number_of_owners = codata.data(all_edges, NUM_OWNERS);
metrics = [codata.data(all_edges, N_PMI) codata.data(all_edges, PMI) codata.data(all_edges, COND_PROB) codata.data(all_edges, MIN_COND_PROB) codata.data(all_edges, MEAN_COND_PROB)];
metric_str = {'Normalized PMI', 'PMI', 'Conditional Probability', 'Minimum Conditional Probability', 'Mean Conditional Probability'};
[x_all, y_all] = ind2sub(codata.dataDims(1:2), all_edges(:));
all_edges_str = [data.concepts(y_all), data.concepts(x_all)];
num_examples = length(all_edges);
num_relationships = size(sorted_relations_str, 1);
cross_valid_ind = crossvalind('KFold', num_examples, 5);

per_class_scores_cubic = zeros(num_examples, num_relationships);
per_class_scores_tree = zeros(num_examples, num_relationships);
per_class_scores_gauss = zeros(num_examples, num_relationships);

all_label = relations(all_edges, :);
[x, y] = find(all_label);
[x_unique, ~, ind_unique] = unique(x);
all_label_str = cell(num_examples, 1);
all_label_ind = cell(num_examples, 1);
for ii =1:num_examples
    all_label_str{ii} = strjoin(sorted_relations_str(y(x==ii)), ';');
    all_label_ind{ii} = [y(x==ii)];
end

%Relationship distribution
relationship_dist = sum(all_label, 1);


if use_diff
    display('Using feature difference');
else
    display('Using concatenated features');
end 

if use_equal_weight
    display('Using equal weights');
else
    display('Using all examples');
end

classifier_relationship_mask = ~ismember(sorted_relations_str, {'AtLocationGeographic', 'GeographicContainment'});
classifier_relationships = find(classifier_relationship_mask);


for fold =1:5
    fold_train_mask = false(size(in_flickr));
    fold_test_mask = false(size(in_flickr));
    fold_train_mask(all_edges(cross_valid_ind~=fold)) = true; 
    fold_test_mask(cross_valid_ind==fold) = true;
    test_str = [data.concepts(y_all(fold_test_mask)), data.concepts(x_all(fold_test_mask))];
    
    [ per_class_scores_cubic, per_class_scores_gauss ] =  train_test_classifiers( codata.dataDims, ...
        word2vec, relations, sorted_relations_str, all_edges, train_mask, test_mask, ...
        per_class_scores_cubic, per_class_scores_gauss, use_diff, use_equal_weight);
end %for

%Test filtered relationships, AtLocationGeographic and GeographicContainment
[ per_class_scores_cubic ] = filter_classifier( all_edges, geographic_containment, ...
    'GeographicContainment', sorted_relations_str, per_class_scores_cubic );
[ per_class_scores_gauss ] = filter_classifier( all_edges, geographic_containment, ...
    'GeographicContainment', sorted_relations_str, per_class_scores_gauss );
% [ per_class_scores_tree ] = filter_classifier( all_edges, geographic_containment, ...
%     'GeographicContainment', sorted_relations_str, per_class_scores_tree );

[ per_class_scores_cubic ] = filter_classifier( all_edges, at_location_geographic, ...
    'AtLocationGeographic', sorted_relations_str, per_class_scores_cubic );
[ per_class_scores_gauss ] = filter_classifier( all_edges, at_location_geographic, ...
    'AtLocationGeographic', sorted_relations_str, per_class_scores_gauss );
% [ per_class_scores_tree ] = filter_classifier( all_edges, at_location_geographic, ...
%     'AtLocationGeographic', sorted_relations_str, per_class_scores_tree );

per_class_scores_cubic = filter_multiclass_output(sorted_relations_str, all_edges_str, per_class_scores_cubic, root, search_description);
per_class_scores_gauss = filter_multiclass_output(sorted_relations_str, all_edges_str, per_class_scores_gauss, root, search_description);
% per_class_scores_tree = filter_multiclass_output(sorted_relations_str, all_edges_str, per_class_scores_tree, root, search_description);

save_str = [search_description '_' save_str];

[cubic_result, cubic_error_rates] = summerize_error(root, save_str, ...
    'CubicSVM', per_class_scores_cubic, all_label, sorted_relations_str, all_edges_str, ...
    all_label_str, number_of_images, number_of_owners, metrics, metric_str, use_threshold);
[gauss_result, gauss_error_rates] = summerize_error(root, save_str, ...
    'GaussSVM', per_class_scores_gauss, all_label, sorted_relations_str, all_edges_str, ...
    all_label_str, number_of_images, number_of_owners, metrics, metric_str, use_threshold);
% [tree_result, tree_error_rates] = summerize_error(root, save_str, ...
%     'BoostedTree', per_class_scores_tree, all_label, sorted_relations_str, all_edges_str, ...
%     all_label_str, number_of_images, number_of_owners, metrics, metric_str, use_threshold);

error_rates = [cubic_error_rates; gauss_error_rates; tree_error_rates];
end %function


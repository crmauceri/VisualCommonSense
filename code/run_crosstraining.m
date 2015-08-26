function [cubic_result, gauss_result, error_rates, edges_per_relation] = ...
    run_crosstraining(root, search_description, use_equal_weight, use_diff, ...
    use_glove, glove_name, use_threshold, use_merged_relations, min_num_images, ...
    min_num_edges, use_approx, save_str, vocab_train, vocab_test, root_train, root_test, ...
    search_description_train, search_description_test)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
addpath('transfer');
addpath('analysis');
addpath('prec_rec');
variable_def;

[has_relations_mask, in_flickr, sorted_relations_str, relations, ...
    geographic_containment, at_location_geographic, ~, ...
    word2vec, ~] = initialize_workspace_retrieve(root, ...
    search_description, use_glove, glove_name, use_merged_relations, ...
    min_num_images, min_num_edges, -1, use_approx);

[has_relations_mask_train, in_flickr_train, ~, ~, ...
    ~, ~, ~, ~, ~] = initialize_workspace_retrieve(root_train, ...
    search_description_train, use_glove, glove_name, use_merged_relations, ...
    min_num_images, min_num_edges, -1, use_approx);

% [has_relations_mask_test, in_flickr_test, ~, ~, ...
%     ~, ~, ~, ~, ~] = initialize_workspace_retrieve(root_test, ...
%     search_description_test, use_glove, glove_name, use_merged_relations, ...
%     min_num_images, min_num_edges, -1, use_approx);

%Flickr data
if use_approx
    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_images.mat', ...
        root, search_description));
else
    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', ...
        root, search_description));
end
data = load(sprintf('%s/Flickr_concepts/%s_analysis.mat', root, search_description));

%Training set
[train_overlap, train_order] = ismember(data.concepts, vocab_train); 
train_overlap = double(train_overlap);
train_insert_mask = (train_overlap*train_overlap')==1;
[x, y] = find(train_insert_mask);
train_insert_ind = sub2ind([length(vocab_train), length(vocab_train)], train_order(x), train_order(y));

new_size = length(data.concepts)^2; 
train_in_flickr_relations_mask = zeros(new_size, 1); 
train_in_flickr_relations_mask(train_insert_mask(:)) = has_relations_mask_train(train_insert_ind) & in_flickr_train(train_insert_ind);

train_mask = double(ismember(data.concepts, vocab_train));
train_mask = train_mask * train_mask';
train_mask = train_mask(:) & train_in_flickr_relations_mask & in_flickr & has_relations_mask;
%Not all the relationships may be in the training set
edges_per_relation = sum(relations(train_mask, :));
[edges_per_relation, sorted_order] = sort(edges_per_relation, 'descend');
relations = relations(:, sorted_order(edges_per_relation>20));
sorted_relations_str = sorted_relations_str(sorted_order(edges_per_relation>20));
num_relationships = size(sorted_relations_str, 1);

%Test set
test_mask = double(ismember(data.concepts, vocab_test));
test_mask = test_mask * test_mask'; 
test_mask = test_mask(:) & ~train_mask;
test_mask = test_mask & in_flickr & has_relations_mask;

%Edges to test
all_edges_ind = find(test_mask);
[x_all, y_all] = ind2sub(codata.dataDims(1:2), all_edges_ind(:));
all_edges_str = [data.concepts(y_all), data.concepts(x_all)];
num_examples = length(all_edges_ind);

per_class_scores_cubic = zeros(num_examples, num_relationships);
per_class_scores_gauss = zeros(num_examples, num_relationships);
% per_class_scores_tree = zeros(num_examples, num_relationships);

number_of_images = codata.data(all_edges_ind, NUM_IMAGES);
number_of_owners = codata.data(all_edges_ind, NUM_OWNERS);
metrics = [codata.data(all_edges_ind, N_PMI), codata.data(all_edges_ind, PMI), ...
    codata.data(all_edges_ind, COND_PROB), codata.data(all_edges_ind, MIN_COND_PROB), ...
    codata.data(all_edges_ind, MEAN_COND_PROB)];
metric_str = {'Normalized PMI', 'PMI', 'Conditional Probability', ...
    'Minimum Conditional Probability', 'Mean Conditional Probability'};

ground_truth_label = relations(all_edges_ind, :);
[x, y] = find(ground_truth_label);
ground_truth_label_str = cell(num_examples, 1);
ground_truth_label_ind = cell(num_examples, 1);
for ii =1:num_examples
    ground_truth_label_str{ii} = strjoin(sorted_relations_str(y(x==ii)), ';');
    ground_truth_label_ind{ii} = y(x==ii);
end

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

%Classification    
[ per_class_scores_cubic, per_class_scores_gauss ] =  train_test_classifiers( codata.dataDims, ...
    word2vec, relations, sorted_relations_str, all_edges_ind, train_mask, test_mask, ...
    per_class_scores_cubic, per_class_scores_gauss, use_diff, use_equal_weight);


%Test filtered relationships, AtLocationGeographic and GeographicContainment
[ per_class_scores_cubic ] = filter_classifier( all_edges_ind, geographic_containment, ...
    'GeographicContainment', sorted_relations_str, per_class_scores_cubic );
[ per_class_scores_gauss ] = filter_classifier( all_edges_ind, geographic_containment, ...
    'GeographicContainment', sorted_relations_str, per_class_scores_gauss );
% [ per_class_scores_tree ] = filter_classifier( all_edges, geographic_containment, ...
%     'GeographicContainment', sorted_relations_str, per_class_scores_tree );

[ per_class_scores_cubic ] = filter_classifier( all_edges_ind, at_location_geographic, ...
    'AtLocationGeographic', sorted_relations_str, per_class_scores_cubic );
[ per_class_scores_gauss ] = filter_classifier( all_edges_ind, at_location_geographic, ...
    'AtLocationGeographic', sorted_relations_str, per_class_scores_gauss );
% [ per_class_scores_tree ] = filter_classifier( all_edges, at_location_geographic, ...
%     'AtLocationGeographic', sorted_relations_str, per_class_scores_tree );

per_class_scores_cubic = filter_multiclass_output(sorted_relations_str, all_edges_str, ...
    per_class_scores_cubic, root, search_description);
per_class_scores_gauss = filter_multiclass_output(sorted_relations_str, all_edges_str, ...
    per_class_scores_gauss, root, search_description);
% per_class_scores_tree = filter_multiclass_output(sorted_relations_str, all_edges_str, ...
%     per_class_scores_tree, root, search_description);

save_str = [search_description '_' save_str];

[cubic_result, cubic_error_rates] = summerize_error(root, save_str, ...
    'CubicSVM', per_class_scores_cubic, ground_truth_label, sorted_relations_str, all_edges_str, ...
    ground_truth_label_str, number_of_images, number_of_owners, metrics, metric_str, use_threshold);
[gauss_result, gauss_error_rates] = summerize_error(root, save_str, ...
    'GaussSVM', per_class_scores_gauss, ground_truth_label, sorted_relations_str, all_edges_str, ...
    ground_truth_label_str, number_of_images, number_of_owners, metrics, metric_str, use_threshold);
% [tree_result, tree_error_rates] = summerize_error(root, save_str, ...
%     'BoostedTree', per_class_scores_tree, all_label, sorted_relations_str, all_edges_str, ...
%     all_label_str, number_of_images, number_of_owners, metrics, metric_str, use_threshold);

error_rates = [cubic_error_rates; gauss_error_rates]; % tree_error_rates];
end %function


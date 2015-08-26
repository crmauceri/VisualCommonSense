function [cubic_result, gauss_result] = ...
    run_classify_unlabeled(root, search_description, use_equal_weight, use_diff, ...
    use_glove, glove_name, use_threshold, use_merged_relations, min_num_images, ...
    min_num_edges, min_images_unlabeled, use_approx, save_str)
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
train_mask = in_flickr & has_relations_mask;

%Test set
labeled_mask = reshape(has_relations_mask, codata.dataDims(1:2));
diag_mask = eye(codata.dataDims(1))==1;
labeled_mask = labeled_mask | labeled_mask' | diag_mask;
test_mask = ~labeled_mask(:) & in_flickr & codata.data(:, NUM_IMAGES)>min_images_unlabeled;

%Edges to test
all_edges_ind = find(test_mask);
[x_all, y_all] = ind2sub(codata.dataDims(1:2), all_edges_ind(:));
all_edges_str = [data.concepts(y_all), data.concepts(x_all)];
num_examples = length(all_edges_ind);
num_relationships = size(sorted_relations_str, 1);

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

sorted_relations_str = [{' '}; sorted_relations_str];
[labels, sorted_score, ~] = get_labels(per_class_scores_cubic, use_threshold);
max_score = zeros(codata.dataDims(1:2));
max_score(all_edges_ind) = sorted_score(:, 1);
max_score = (max(max_score, max_score')==max_score) & max_score>0;

labels = labels + 1;
cubic_result = [[{'Tag1', 'Tag2', 'Labeled Relationship1', 'Score1', ...
    'Labeled Relationship2', 'Score2', 'Labeled Relationship3', 'Score3', ...
    'Number of Images', 'Number of Owners'}, metric_str, {'MaxScore'}]; ...
    all_edges_str, ...
    sorted_relations_str(labels(:,1)), num2cell(full(sorted_score(:, 1))), ...
    sorted_relations_str(labels(:,2)), num2cell(full(sorted_score(:, 2))), ...
    sorted_relations_str(labels(:,3)), num2cell(full(sorted_score(:, 3))), ...
    num2cell(full(number_of_images)), ...
    num2cell(full(number_of_owners)), ...
    num2cell(full(metrics)), ...
    num2cell(max_score(all_edges_ind))
    ];

[labels, sorted_score, ~] = get_labels(per_class_scores_gauss, use_threshold);
labels = labels + 1;
gauss_result = [[{'Tag1', 'Tag2', 'Labeled Relationship1', 'Score1', ...
    'Labeled Relationship2', 'Score2', 'Labeled Relationship3', 'Score3', ...
    'Number of Images', 'Number of Owners'}, metric_str]; ...
    all_edges_str, ...
    sorted_relations_str(labels(:,1)), num2cell(full(sorted_score(:, 1))), ...
    sorted_relations_str(labels(:,2)), num2cell(full(sorted_score(:, 2))), ...
    sorted_relations_str(labels(:,3)), num2cell(full(sorted_score(:, 3))), ...
    num2cell(full(number_of_images)), ...
    num2cell(full(number_of_owners)), ...
    num2cell(full(metrics))
    ];
end %function

function [gauss_result, cubic_result] = run_retrieve_multiclass(root, ... 
    search_description, use_equal_weight, use_diff, use_glove, glove_name, ...
    use_threshold, use_merged_relations, min_num_images, min_num_edges, min_images_unlabeled, use_approx)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
addpath('transfer');
addpath('analysis');
variable_def;

[has_relations_mask, in_flickr, sorted_relations_str, relations, ...
    geographic_containment, at_location_geographic, geographic_adjective, ...
    word2vec, edges_per_relation] = initialize_workspace_retrieve(root, ... 
    search_description, use_glove, glove_name, ...
    use_merged_relations, min_num_images, min_num_edges, -1, use_approx);

%Flickr data
if use_approx
    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_images.mat', root, search_description));
else
    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', root, search_description));
end
data = load(sprintf('%s/Flickr_concepts/%s_analysis.mat', root, search_description));

% The edges we want to label
unlabeled_mask = reshape(has_relations_mask, codata.dataDims(1:2));
diag_mask = eye(codata.dataDims(1))==1;
unlabeled_mask = unlabeled_mask | unlabeled_mask' | diag_mask;
unlabeled_edges = find(~unlabeled_mask(:) & in_flickr);
number_of_images = codata.data(unlabeled_edges, NUM_IMAGES);
number_of_owners = codata.data(unlabeled_edges, NUM_OWNERS);
unlabeled_edges = unlabeled_edges(number_of_images>min_images_unlabeled);
number_of_owners = number_of_owners(number_of_images>min_images_unlabeled);
number_of_images = number_of_images(number_of_images>min_images_unlabeled);
num_test = length(unlabeled_edges);
metrics = [codata.data(unlabeled_edges, N_PMI) codata.data(unlabeled_edges, PMI) codata.data(unlabeled_edges, COND_PROB) codata.data(unlabeled_edges, MIN_COND_PROB) codata.data(unlabeled_edges, MEAN_COND_PROB)];
metric_str = {'Normalized PMI', 'PMI', 'Conditional Probability', 'Minimum Conditional Probability', 'Mean Conditional Probability'};
[x_un, y_un] = ind2sub(codata.dataDims(1:2), unlabeled_edges(:));
unlabeled_edges_str = [data.concepts(y_un), data.concepts(x_un)];
per_class_scores_cubic = -Inf*ones(size(unlabeled_edges,1), size(sorted_relations_str, 1));
per_class_scores_gauss = -Inf*ones(size(unlabeled_edges,1), size(sorted_relations_str, 1));

if use_diff
    display('Using feature difference');
    unlabeled_feature = (word2vec.features(x_un, :) - word2vec.features(y_un, :));
else
    display('Using concatenated features');
    unlabeled_feature = [word2vec.features(x_un, :), word2vec.features(y_un, :)];
end 


if use_equal_weight
    display('Using equal weights');
else
    display('Using all examples');
end

%Train and test all classifiers
classifier_relationships = find(~ismember(sorted_relations_str, {'AtLocationGeographic', 'GeographicContainment'}));
for relation_ind = classifier_relationships'
    relationship_mask = spones(relations(:,relation_ind));

    %Simple example, use one relationship as positive, other as negatives
    positive_edges = find(relationship_mask & in_flickr);
    negative_edges = find(~relationship_mask & has_relations_mask & in_flickr);

    if length(positive_edges)>20  
        [x_pos, y_pos] = ind2sub(codata.dataDims(1:2), positive_edges(:));
        [x_neg, y_neg] = ind2sub(codata.dataDims(1:2), negative_edges(:));

        positive_edges_str = [data.concepts(y_pos), data.concepts(x_pos)];
        if use_diff
            %display('Using feature difference');
            positive_feature = (word2vec.features(x_pos, :)- word2vec.features(y_pos, :));
            negative_feature = (word2vec.features(x_neg, :) - word2vec.features(y_neg, :));
            
        else
            %display('Using concatenated features');
            positive_feature = [word2vec.features(x_pos, :), word2vec.features(y_pos, :)];
            negative_feature = [word2vec.features(x_neg, :), word2vec.features(y_neg, :)];
        end

        %Final processing for training set
        num_pos = size(positive_feature, 1);
        num_neg = size(negative_feature, 1);
        %display(sprintf('Number of positive examples: %d', num_pos));
        %display(sprintf('Number of negative examples: %d', num_neg));
        num_pos_neg = [num_pos, num_neg];
        if use_equal_weight
            %display('Using equal weights');
            neg_select = randperm(num_neg, num_pos);
            feature = [positive_feature; negative_feature(neg_select,:)];
            negative_edges_str = negative_edges_str(neg_select);
            num_neg = num_pos;
        else
            %display('Using all examples');
            feature = [positive_feature; negative_feature];
        end
        num_examples = num_pos + num_neg;
        class = [ones(num_pos, 1); zeros(num_neg, 1)];
        order = randperm(num_examples);
        feature = feature(order, :);
        class = class(order);

        SVMModel = fitcsvm(feature, class, ...
            'KernelFunction','polynomial', 'KernelScale','auto', ...
            'PolynomialOrder', 3, 'Standardize',true);
        [~, score] = predict(SVMModel, unlabeled_feature);
        per_class_scores_cubic(:,relation_ind) = score(:, 2);
        
        SVMModel = fitcsvm(feature, class, ...
            'KernelFunction','rbf', 'KernelScale', 27, ...
            'BoxConstraint', 100, 'Standardize',true);
        [~, score] = predict(SVMModel, unlabeled_feature);
        per_class_scores_gauss(:,relation_ind) = score(:,2);
    end %if

end%for

%Test filter relationships, AtLocationGeographic and GeographicContainment
geo_containment_ind = find(ismember(sorted_relations_str, 'GeographicContainment'));
geo_containment_edge_mask = ismember(unlabeled_edges, find(geographic_containment(:)));
per_class_scores_cubic(geo_containment_edge_mask ,geo_containment_ind) = Inf;
per_class_scores_cubic(~geo_containment_edge_mask ,geo_containment_ind) = -Inf;
per_class_scores_gauss(geo_containment_edge_mask ,geo_containment_ind) = Inf;
per_class_scores_gauss(~geo_containment_edge_mask ,geo_containment_ind) = -Inf;

location_geo_ind = find(ismember(sorted_relations_str, 'AtLocationGeographic'));
location_geo_edge_mask = ismember(unlabeled_edges, find(at_location_geographic(:)));
per_class_scores_cubic(location_geo_edge_mask, location_geo_ind) = Inf;
per_class_scores_cubic(~location_geo_edge_mask, location_geo_ind) = -Inf;
per_class_scores_gauss(location_geo_edge_mask, location_geo_ind) = Inf;
per_class_scores_gauss(~location_geo_edge_mask, location_geo_ind) = -Inf;

per_class_scores_cubic = filter_multiclass_output(sorted_relations_str, unlabeled_edges_str, per_class_scores_cubic, root, search_description);
per_class_scores_gauss = filter_multiclass_output(sorted_relations_str, unlabeled_edges_str, per_class_scores_gauss, root, search_description);

sorted_relations_str = [{' '}; sorted_relations_str];
[labels, sorted_score, ~] = get_labels(per_class_scores_cubic, use_threshold);
labels = labels + 1;
cubic_result = [[{'Tag1', 'Tag2', 'Labeled Relationship1', 'Score1', ...
    'Labeled Relationship2', 'Score2', 'Labeled Relationship3', 'Score3', ...
    'Number of Images', 'Number of Owners'}, metric_str]; ...
    unlabeled_edges_str, ...
    sorted_relations_str(labels(:,1)), num2cell(full(sorted_score(:, 1))), ...
    sorted_relations_str(labels(:,2)), num2cell(full(sorted_score(:, 2))), ...
    sorted_relations_str(labels(:,3)), num2cell(full(sorted_score(:, 3))), ...
    num2cell(full(number_of_images)), ...
    num2cell(full(number_of_owners)), ...
    num2cell(full(metrics))
    ];


[labels, sorted_score, ~] = get_labels(per_class_scores_gauss, use_threshold);
labels = labels + 1;
gauss_result = [[{'Tag1', 'Tag2', 'Labeled Relationship1', 'Score1', ...
    'Labeled Relationship2', 'Score2', 'Labeled Relationship3', 'Score3', ...
    'Number of Images', 'Number of Owners'}, metric_str]; ...
    unlabeled_edges_str, ...
    sorted_relations_str(labels(:,1)), num2cell(full(sorted_score(:, 1))), ...
    sorted_relations_str(labels(:,2)), num2cell(full(sorted_score(:, 2))), ...
    sorted_relations_str(labels(:,3)), num2cell(full(sorted_score(:, 3))), ...
    num2cell(full(number_of_images)), ...
    num2cell(full(number_of_owners)), ...
    num2cell(full(metrics))
    ];
end %function


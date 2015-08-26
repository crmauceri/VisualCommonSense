function [gauss_result, cubic_result, error_rates] = run_retrieve_multiclass(root, ... 
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
rooms_root = 'E:/rooms/data';
animals_root = 'E:\data\Iconic\data';
search_description_animals = 'biology_domesticated_animal_500_extended';
search_description_rooms = 'rooms_200_extended';
feature_dir = sprintf('%s\\word2vec_features\\', rooms_root);
rooms_feature_path = [feature_dir search_description_rooms '_feature_' glove_name '.mat'];
feature_dir = sprintf('%s\\word2vec_features\\', animals_root);
animals_feature_path = [feature_dir search_description_animals '_feature_' glove_name '.mat'];

rooms_word2vec = load(rooms_feature_path);
rooms_word2vec.concepts = cellstr(rooms_word2vec.concepts);
animals_word2vec = load(animals_feature_path);
animals_word2vec.concepts = cellstr(animals_word2vec.concepts);

training_mask = double(ismember(data.concepts, rooms_word2vec.concepts));
training_mask = training_mask * training_mask';
test_mask = double(ismember(data.concepts, animals_word2vec.concepts));
test_mask = test_mask * test_mask'; 
test_mask = test_mask & ~training_mask;
test_mask = test_mask(:) & in_flickr & has_relations_mask;

%The edges we want to label for cross fold validation error
all_edges = find(test_mask);
number_of_images = codata.data(all_edges, NUM_IMAGES);
number_of_owners = codata.data(all_edges, NUM_OWNERS);
metrics = [codata.data(all_edges, N_PMI) codata.data(all_edges, PMI) codata.data(all_edges, COND_PROB) codata.data(all_edges, MIN_COND_PROB) codata.data(all_edges, MEAN_COND_PROB)];
metric_str = {'Normalized PMI', 'PMI', 'Conditional Probability', 'Minimum Conditional Probability', 'Mean Conditional Probability'};
[x_all, y_all] = ind2sub(codata.dataDims(1:2), all_edges(:));
all_edges_str = [data.concepts(y_all), data.concepts(x_all)];
num_examples = length(all_edges);
cross_valid_ind = crossvalind('KFold', num_examples, 5);

number_training_relationship = sum(relations(in_flickr & training_mask(:), :));
[number_training_relationship, sorted_order] = sort(number_training_relationship, 'descend');
relations = relations(:, sorted_order(number_training_relationship>20));
sorted_relations_str = sorted_relations_str(sorted_order(number_training_relationship>20));
num_relationships = size(sorted_relations_str, 1);

per_class_scores_cubic = zeros(num_examples, num_relationships);
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

[x_un, y_un] = ind2sub(codata.dataDims(1:2), all_edges);
per_class_scores_cubic = -Inf*ones(num_examples, num_relationships);
per_class_scores_gauss = -Inf*ones(num_examples, num_relationships);

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
    positive_edges = find(relationship_mask & in_flickr & training_mask(:));
    negative_edges = find(~relationship_mask & has_relations_mask & in_flickr & training_mask(:));

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
        display(sprintf('Number of positive examples: %d', num_pos));
        display(sprintf('Number of negative examples: %d', num_neg));
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
geo_containment_edge_mask = ismember(all_edges, find(geographic_containment(:)));
per_class_scores_cubic(geo_containment_edge_mask ,geo_containment_ind) = 1;
per_class_scores_cubic(~geo_containment_edge_mask ,geo_containment_ind) = -1;
per_class_scores_gauss(geo_containment_edge_mask ,geo_containment_ind) = 1;
per_class_scores_gauss(~geo_containment_edge_mask ,geo_containment_ind) = -1;

location_geo_ind = find(ismember(sorted_relations_str, 'AtLocationGeographic'));
location_geo_edge_mask = ismember(all_edges, find(at_location_geographic(:)));
per_class_scores_cubic(location_geo_edge_mask, location_geo_ind) = 1;
per_class_scores_cubic(~location_geo_edge_mask, location_geo_ind) = -1;
per_class_scores_gauss(location_geo_edge_mask, location_geo_ind) = 1;
per_class_scores_gauss(~location_geo_edge_mask, location_geo_ind) = -1;

per_class_scores_cubic = filter_multiclass_output(sorted_relations_str, all_edges_str, per_class_scores_cubic, root, search_description);
per_class_scores_gauss = filter_multiclass_output(sorted_relations_str, all_edges_str, per_class_scores_gauss, root, search_description);

save_str = [search_description '_crosstraining_train_rooms'];

[cubic_result, cubic_error_rates] = summerize_error(root, save_str, ...
    'CubicSVM', per_class_scores_cubic, all_label, sorted_relations_str, all_edges_str, ...
    all_label_str, number_of_images, number_of_owners, metrics, metric_str, use_threshold);
[gauss_result, gauss_error_rates] = summerize_error(root, save_str, ...
    'GaussSVM', per_class_scores_gauss, all_label, sorted_relations_str, all_edges_str, ...
    all_label_str, number_of_images, number_of_owners, metrics, metric_str, use_threshold);

error_rates = [cubic_error_rates; gauss_error_rates];
end %function


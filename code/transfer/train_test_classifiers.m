function [ per_class_scores_cubic, per_class_scores_gauss ] = ...
    train_test_classifiers( dataDims, word2vec, relations_matrix, ...
    relations_list, all_edge_ind, train_mask, test_mask, per_class_scores_cubic, ...
    per_class_scores_gauss, use_diff, use_equal_weight)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

all_edges_mask = ismember(all_edge_ind, find(test_mask));

[x_test, y_test] = find(reshape(test_mask, dataDims(1:2)));
if use_diff
    test_feature = (word2vec.features(x_test, :) - word2vec.features(y_test, :));
else
    test_feature = [word2vec.features(x_test, :), word2vec.features(y_test, :)];
end

classifier_relationship_mask = ~ismember(relations_list, {'AtLocationGeographic', 'GeographicContainment'});
classifier_relationships = find(classifier_relationship_mask);

for relation_ind = classifier_relationships'
        relationship_mask = spones(relations_matrix(:,relation_ind));

        %Simple example, use one relationship as positive, other as negatives
        positive_edges = find(relationship_mask & train_mask);
        negative_edges = find(~relationship_mask & train_mask);

        %if length(positive_edges)>20  
        [x_pos, y_pos] = ind2sub(dataDims(1:2), positive_edges(:));
        [x_neg, y_neg] = ind2sub(dataDims(1:2), negative_edges(:));

        if use_diff
            display('Using feature difference');
            positive_feature = (word2vec.features(x_pos, :)- word2vec.features(y_pos, :));
            negative_feature = (word2vec.features(x_neg, :) - word2vec.features(y_neg, :));
            
        else
            display('Using concatenated features');
            positive_feature = [word2vec.features(x_pos, :), word2vec.features(y_pos, :)];
            negative_feature = [word2vec.features(x_neg, :), word2vec.features(y_neg, :)];
        end

        %Final processing for training set
        num_pos = size(positive_feature, 1);
        num_neg = size(negative_feature, 1);
         display(sprintf('Number of positive examples: %d', num_pos));
         display(sprintf('Number of negative examples: %d', num_neg));

        if use_equal_weight
            neg_select = randperm(num_neg, num_pos);
            feature = [positive_feature; negative_feature(neg_select,:)];
            negative_edges_str = negative_edges_str(neg_select);
            num_neg = num_pos;
        else
            feature = [positive_feature; negative_feature];
        end
        num_train_examples = num_pos + num_neg;
        class = [ones(num_pos, 1); zeros(num_neg, 1)];
        order = randperm(num_train_examples);
        feature = feature(order, :);
        class = class(order);

        SVMModel = fitcsvm(feature, class, ...
            'KernelFunction','polynomial', 'KernelScale','auto', ...
            'PolynomialOrder', 3, 'Standardize',true);
        [~, score] = predict(SVMModel, test_feature);
       per_class_scores_cubic(all_edges_mask, relation_ind) = score(:, 2);

        SVMModel = fitcsvm(feature, class, ...
            'KernelFunction','rbf', 'KernelScale', 27, ...
            'BoxConstraint', 100, 'Standardize',true);
        [~, score] = predict(SVMModel, test_feature);
        per_class_scores_gauss(all_edges_mask, relation_ind) = score(:,2);
        %end %if

end%for
    
%     %Decision Tree
%     nlabels = cellfun(@length, all_label_ind(cross_valid_ind~=fold))';
%     replabel_ind = zeros(1, sum(nlabels));
%     replabel_ind(cumsum([1 nlabels(1:end-1)])) = 1;
%     replabel_ind = cumsum(replabel_ind); 
%     feature_ind = find(fold_train_mask);
%     feature_ind = feature_ind(replabel_ind);
%     [x_train, y_train] = ind2sub(dataDims(1:2), feature_ind);
%     
%     feature_ind_test = find(fold_test_mask);
%     [x_test, y_test] = ind2sub(dataDims(1:2), feature_ind_test);
%     if use_diff
%         display('Using feature difference');
%         feature = (word2vec.features(x_train, :)- word2vec.features(y_train, :));
%         test_feature = (word2vec.features(x_test, :)- word2vec.features(y_test, :));
%     else
%         display('Using concatenated features');
%         feature = [word2vec.features(x_train, :), word2vec.features(y_train, :)];
%         test_feature = [word2vec.features(x_test, :), word2vec.features(y_test, :)];
%     end
%     
%     class = cell2mat(all_label_ind(cross_valid_ind~=fold));
%     
%     TreeModel = fitensemble(feature, class, ...
%         'AdaBoostM2', 100, 'Tree');
%     [~, scores] = predict(TreeModel, test_feature);
%     per_class_scores_tree(fold_test_mask, classifier_relationship_mask) = scores(:, classifier_relationship_mask);
        

end


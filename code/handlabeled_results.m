label_save_str = sprintf('%s/validation/multi_class_%s_balanced_random_processed.mat', root, search_description);
hand_labeled = load(label_save_str);

max_score = zeros(codata.dataDims(1:2));
max_score(all_edges_ind) = max(per_class_scores_cubic, [], 2);
max_score_undirected = max(max_score, max_score');
is_max_direction = max_score == max_score_undirected;

[relationship_mask, relationship_order] = ismember(sorted_relations_str, hand_labeled.relationships);
hand_labeled_mask = any(hand_labeled.edge_label(:, relationship_order(relationship_mask)), 2);
hand_labeled_ind = find(hand_labeled.edge_label_mask(:)&test_mask);
[hand_labeled_keep, hand_labeled_order] = ismember(all_edges_ind, hand_labeled_ind);

ground_truth_label_str = cell(size(x));
ground_truth_label = hand_labeled.edge_label(hand_labeled_ind, relationship_order(relationship_mask));
for ii = 1:length(sorted_relations_str)
   edges_with_relationship = ground_truth_label(:, ii) > 0;
   ground_truth_label_str(edges_with_relationship) = repmat(sorted_relations_str(ii), nnz(edges_with_relationship), 1);
end
ground_truth_label = [ground_truth_label, ~any(ground_truth_label, 2)];

unlabeled_edges = ~ismember(all_edges_ind(hand_labeled_keep), find(is_max_direction));
unlabeled_edges_score = zeros(size(unlabeled_edges));
unlabeled_edges_score(unlabeled_edges) = Inf;
per_class_scores_cubic_labeled = [per_class_scores_cubic(hand_labeled_keep, :), unlabeled_edges_score];
all_edges_str_labeled = all_edges_str(hand_labeled_keep, :);
number_of_images_labeled = number_of_images(hand_labeled_keep);
number_of_owners_labeled = number_of_owners(hand_labeled_keep);
metrics_labeled = metrics(hand_labeled_keep, :);

save_str = [search_description 'hand_labeled'];
[cubic_result, cubic_error_rates] = summerize_error(root, save_str, ...
    'CubicSVM', per_class_scores_cubic_labeled, ground_truth_label, [sorted_relations_str; {'Unlabeled'}], all_edges_str_labeled, ...
    ground_truth_label_str, number_of_images_labeled, number_of_owners_labeled, metrics_labeled, metric_str, use_threshold);

header = {'Classifier', 'Accuracy', 'Recall' , 'Precision', 'Recall@3', 'Precision@3'};
    error_rates_result = [header; [{'Cubic'}, num2cell(cubic_error_rates)]];
    save_path = sprintf('%s/transfer/%s_%s_multi_class_error.txt', root, save_str);
    cell2csv(save_path, error_rates_result, '\t');

% combined_save = sprintf('%s/transfer/%s_result.txt', root, save_str);
% cell2csv(combined_save, cubic_result, '\t');
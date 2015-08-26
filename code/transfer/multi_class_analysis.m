function [recall, precision, accuracy, misclassified, misclassified_3, unlabeled, labels, ...
    sorted_score] = multi_class_analysis(multi_class_scores, ... 
    multi_class_ground_truth, use_threshold, relationships, save_path)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


[num_examples, num_classes] = size(multi_class_scores);
[labels, sorted_score, unlabeled] = get_labels(multi_class_scores, use_threshold);

%True Positives
label_mask_at_one = zeros(num_examples, num_classes);
label_mask_at_one(sub2ind(size(label_mask_at_one), 1:num_examples, labels(:, 1)')) = 1;
accuracy = full(sum(sum(label_mask_at_one & multi_class_ground_truth, 1))/num_examples);
misclassified = ~any(label_mask_at_one & multi_class_ground_truth, 2);

label_mask_at_three = zeros(num_examples, num_classes);
label_mask_at_three(sub2ind(size(label_mask_at_three), repmat(1:num_examples, 3, 1), labels(:, 1:3)')) = 1;
misclassified_3 = ~any(label_mask_at_three & multi_class_ground_truth, 2);

%Confusion Matrix
confusion = zeros(num_classes, num_classes);
for ii = 1:num_examples
    if any(label_mask_at_one(ii, :) & multi_class_ground_truth(ii, :))
        label = labels(ii, 1);
        confusion(label, label) = confusion(label, label) + 1;
    else
        confusion = confusion + (multi_class_ground_truth(ii, :)' * label_mask_at_one(ii, :));
    end
end
num_per_class = sum(multi_class_ground_truth, 1);
%confusion = log(confusion); %./repmat(num_per_class, num_classes, 1)';

axis_labels = cellfun(@(x) x(1:min(length(x), 25)), relationships, 'UniformOutput', false);
h = display_confusion(confusion, axis_labels);
c = colorbar();
c.TickLabels = round(exp(c.Ticks));
saveas(h, sprintf('%s_confusion.fig', save_path));

%Precision-Recall
recall = zeros(num_classes, num_classes);
precision = zeros(num_classes, num_classes);
for ii=1:num_classes
    positives = zeros(num_examples, num_classes);
    positives(sub2ind(size(label_mask_at_one), repmat(1:num_examples, ii, 1), labels(:, 1:ii)')) = 1;
    true_postives = sum(positives & multi_class_ground_truth, 1);

    recall(ii, :) = true_postives ./ sum(multi_class_ground_truth, 1);
    precision(ii, :) = true_postives ./ sum(positives, 1);
end

h = figure;
subplot(1,5, 1:2); hold on;
for ii=1:num_classes
    if ~all(multi_class_scores(:, ii) == multi_class_scores(1, ii))
        prec_rec(multi_class_scores(:, ii), multi_class_ground_truth(:, ii), 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1);
    end
end
xlabel('Precision');
ylabel('Recall');
set(findall(gca, 'Type', 'Line'),'LineWidth',1.2);
title('Precision Recall Curve');

%ROC
subplot(1,5, 3:5); hold on;
[tpr, fpr, ~] = roc(multi_class_ground_truth', multi_class_scores');
for ii=1:num_classes  
    plot(fpr{ii}, tpr{ii});
end
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend(relationships, 'Location', 'eastoutside');
set(findall(gca, 'Type', 'Line'),'LineWidth',1.2);
title('ROC Curve');
saveas(h, sprintf('%s_roc_pr.fig', save_path));
end


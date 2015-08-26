function [labels, sorted_score, unlabeled] = get_labels(multi_class_scores, use_threshold)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[num_examples, ~] = size(multi_class_scores);

[sorted_score, labels] = sort(multi_class_scores, 2, 'descend');
if use_threshold
    labels(sorted_score<=0) = 0; %Negative scores are unlabeled. 
    unlabeled = sorted_score(:,1)<=0;
else
    unlabeled = false(num_examples, 1);
end

end


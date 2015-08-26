function [ relations, relation_str ] = merge_relations( relations, relation_str, relations_to_merge, merge_order, new_relationship_name, shape )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

expanded_relations = {}; % cellfun(@(x) strsplit(x, ','), relation_str);
expanded_ind = [];

for ii=1:length(relation_str)
   split = strsplit(relation_str{ii}, ',');
   expanded_relations = [expanded_relations split];
   expanded_ind = [expanded_ind, ii*ones(size(split))];
end

[ind, order] = unique(expanded_ind(ismember(expanded_relations, relations_to_merge)));
if ~isempty(ind)
   merged_relation = relations(:, ind);
   flipped_ind = find(ismember(merge_order(order), -1));
   for ii=flipped_ind 
      merged_relation(:, ii) = reshape(transpose(reshape(merged_relation(:, ii), shape(1:2))), shape(1)*shape(2), []);   
   end
   
   merged_relation = spones(sum(merged_relation, 2));
   relations(:, ind(1)) = merged_relation;
   relation_str{ind(1)} = new_relationship_name;
   
   ind(1) = [];
   relations(:, ind) = [];
   relation_str(ind) = [];
end


end


function [ relations ] = make_disjoint_relations(relations, relation_str, primary_relation, secondary_relations)
%MAKE_DISJOINT_RELATIONS Removes any elements from secondary_relations
%which are also in primary_relation
%   Detailed explanation goes here

primary_ind = ismember(relation_str, primary_relation);
secondary_ind = ismember(relation_str, secondary_relations);

if any(primary_ind) && any(secondary_ind)
    relations(:, secondary_ind) = relations(:, secondary_ind) & ~any(relations(:, primary_ind), 2);
end
end


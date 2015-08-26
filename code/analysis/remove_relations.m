function [ relations, relation_str ] = remove_relations( relations, relation_str, relations_to_remove)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

ind = ismember(relation_str, relations_to_remove);
if (any(ind))
   relations(:, ind) = [];
   relation_str(ind) = [];
end


end


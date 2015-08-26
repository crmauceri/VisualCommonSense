function [ score_matrix ] = filter_classifier( edge_ind, filter_mask, relationship_str, relationship_list, score_matrix )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

relationship_str_ind = find(ismember(relationship_list, relationship_str));
filter_edge_mask = ismember(edge_ind, find(filter_mask(:)));
score_matrix(filter_edge_mask ,relationship_str_ind) = Inf;
score_matrix(~filter_edge_mask ,relationship_str_ind) = -Inf;

end


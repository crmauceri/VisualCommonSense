function [ relations, relation_str ] = split_relations(data, relations, ...
    relation_str, relations_to_filter, concept1_split_filter, concept2_split_filter, ...
    new_relationship_name_with_feature, new_relationship_name_without_feature)
%SPLIT_RELATIONS Split relation_to_filter into 2 relationships on
%split_feature
%   Detailed explanation goes here

relation_ind = ismember(relation_str, relations_to_filter);
if (any(relation_ind))
   num_concepts = size(data.data, 1);
     
   feature1_ind = ismember(data.header, concept1_split_filter);
   feature2_ind = ismember(data.header, concept2_split_filter);
   if(any(feature1_ind))
       tag1_mask = any(data.data(:, feature1_ind), 2); 
   else
       tag1_mask = ones(num_concepts, 1);
   end
   
   if(any(feature2_ind))
       tag2_mask = any(data.data(:, feature2_ind), 2); 
   else
       tag2_mask = ones(num_concepts, 1);
   end
   
   slice_mask = double(tag1_mask) * double(tag2_mask)';
      
   new_relations = zeros(num_concepts*num_concepts, sum(relation_ind)*2);
   jj = 1;
   for ii=find(relation_ind)
       relation_slice = reshape(relations(:, ii), num_concepts, num_concepts);

       relation_slice_with = relation_slice & slice_mask==1;
       relation_slice_without = relation_slice & ~slice_mask==1;
       new_relations(:, jj) = relation_slice_with(:);
       new_relations(:, jj+1) = relation_slice_without(:);
       jj = jj + 2;
   end

   new_relation_str = [new_relationship_name_with_feature; new_relationship_name_without_feature]';

   relations(:, relation_ind) = [];
   relation_str(relation_ind) = [];

   relations = [relations new_relations];
   relation_str = [relation_str; new_relation_str(:)];

end

end


function [relations] = filter_pos_relations(data, relations, relation_str, ...
    relations_to_filter, concept1_feature, concept1_exclude, concept2_feature, concept2_exclude)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

relation_mask = ismember(relation_str, relations_to_filter);
if (any(relation_mask))
   num_concepts = size(data.data, 1);
   
   
   feature1_ind = ismember(data.header, concept1_feature);
   feature2_ind = ismember(data.header, concept2_feature);
   if(any(feature1_ind))
      if(concept1_exclude)
          if isempty(nonzeros(feature1_ind(1:11)))
            %The filter is not for a part of speech
            concept1_mask = ~data.data(:, feature1_ind);
          else
            %The filter is for a part of speech
            pos_filter = [~feature1_ind(1:11), false(1, 4)];
            concept1_mask = any(data.data(:, pos_filter), 2); 
          end
      else
        concept1_mask = data.data(:, feature1_ind); 
      end
   else
       concept1_mask = true(size(data.data, 1), 1);
   end
   
   if(any(feature2_ind))
        if(concept2_exclude)
            if isempty(nonzeros(feature2_ind(1:11)))
                %The filter is not for a part of speech
                concept2_mask = ~data.data(:, feature2_ind);
            else
                %The filter is for a part of speech
                pos_filter = [~feature2_ind(1:11), false(1, 4)];
                concept2_mask = any(data.data(:, pos_filter), 2); 
            end
        else
            concept2_mask = data.data(:, feature2_ind); 
        end
   else
       concept2_mask = true(num_concepts, 1);
   end
   
   relation_ind = find(relation_mask)';
   for ii=relation_ind
       relation_slice = reshape(relations(:, ii), num_concepts, num_concepts);
       slice_mask = repmat(concept1_mask, 1, num_concepts) & repmat(concept2_mask', num_concepts, 1);
       relation_slice = relation_slice & slice_mask';
       relations(:, ii) = relation_slice(:);
   end
end


end


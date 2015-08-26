function [plural_count, cooccurrence_plurals] = add_plurals_to_singular( lemma_mask, cooccurrence_matrix, vocab_count)
%ADD_PLURALS_TO_SINGULAR Add the plural counts of words to the singulars
%   Both lemma_mask and cooccurrence_matrix should be nxn where n is the
%   length of the vocabulary
%   Lemma_mask should be a directed adjacancy matrix without symetric edges
%   between concepts.

transform = lemma_mask + eye(size(lemma_mask));

%Percollate the lemmas, 
%For example "dog" should get the counts for both "puppy" and "puppies".
%Even if there are only edges for "dog"->"puppy" and "puppy"->"puppies"
last_transform = zeros(size(transform));
while any(any(last_transform ~= transform))
   last_transform = transform;
   transform = double((transform * transform) > 0);
end

plural_count =  transform*vocab_count;

%Remove diagonal from cooccurrence_matrix
cooccurrence_matrix(eye(size(cooccurrence_matrix))==1) = 0;
cooccurrence_plurals = transform*cooccurrence_matrix*transform';
cooccurrence_plurals(eye(size(cooccurrence_matrix))==1) = plural_count;
end


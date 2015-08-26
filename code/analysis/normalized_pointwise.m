%% Normalized pointwise mutual information
% pmi = log( p(x,y)/(p(x)*p(y) )/ -log(p(x,y)
%

function A_pmi = normalized_pointwise(A, totalPerRow, total)
[w, ~] = size(totalPerRow);

%Use Laplace smoothing
%Pretend we have one image with all the tags
A = A + 1;
totalPerRow = totalPerRow + 1;
total = total + 1;

pxy = A / total;
px = totalPerRow / total;
px = repmat(px, 1, w);
py = (totalPerRow / total)'; 
py = repmat(py, w, 1);

denominator = px .* py;
A_pmi = log2(pxy ./ denominator)./ -log2(pxy);
end
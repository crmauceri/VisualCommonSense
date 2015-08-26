%% Normalized pointwise mutual information
% pmi = log( p(x,y)/(p(x)*p(y) )/ -log(p(x,y)
%

function A_pmi = pointwise_bound(A, totalPerRow, total)
[w, ~] = size(totalPerRow);

%Use Laplace smoothing
%Pretend we have one image with all the tags
A = A + 1;
totalPerRow = totalPerRow + 1;
total = total + 1;

px = totalPerRow / total;
px = repmat(px, 1, w);
py = (totalPerRow / total)'; 
py = repmat(py, w, 1);

A_pmi = min(-log2(px), -log2(py));
end
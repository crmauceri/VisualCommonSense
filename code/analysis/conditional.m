%% Conditional Probability
% P(x|y) = P(x, y)/P(y)
%
% Different methods
% mean (P(x|y), P(y|x))
% min  (P(x|y), P(y|x))
% direct P(y|x) 

function A = conditional (A, totalPerRow, total, method)
[w, h] = size(totalPerRow);

%Use Laplace smoothing
%Pretend we have one image with all the tags
A = A + 1;
totalPerRow = totalPerRow + 1;
total = total + 1;

pjoint = A / total;
py = totalPerRow / total;
pxy = pjoint ./ repmat(py, 1, w);
pyx = pxy';
    
switch method
    case 'min'
        A = min(pxy, pyx);
    case 'mean'
        A = mean(cat(3, pxy, pyx), 3);
    case 'direct'
        A = pyx; %p(target|source)
end 

end
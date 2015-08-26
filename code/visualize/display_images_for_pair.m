function display_images_for_pair(dataroot, query, tag1, tag2, postfix, offsets)
%DISPLAY_IMAGES_FOR_PAIR Summary of this function goes here
%   Detailed explanation goes here

search_descriptor = query;

load(sprintf('%s/concepts/%s_%s.mat', dataroot, search_descriptor, postfix));
tags = cellstr(concepts);

[~, tag1Ind] = ismember(tag1, tags);
[~, tag2Ind] = ismember(tag2, tags);

%Subtract 1 because of 1 based indexing in Matlab
sortedInd = sort([tag1Ind, tag2Ind]-1, 'descend');

display('Load index')
tic;
pair_index = load_image_pair_index(dataroot, query, sortedInd(1));
toc;

command = sprintf('images = pair_index.s%d;', sortedInd(2));
eval(command);

for offset=offsets
    max_ind = min(25, length(images)-offset);
    if max_ind > 0 
        allI = zeros(200, 200, 3, max_ind);
        curimages = images(offset+1:offset+max_ind);
        image_ids = cellfun(@(x) regexp(x, '/\w*_', 'match'), curimages, 'UniformOutput', false); 

        display('Load images');
        tic;
        index = 1:max_ind;
%         curimages = images;
%         index  = [4 6 9 10 12 14 17 24 25 26 27 28 32 33 37 40];
         for ii=1:max_ind
%         for ii=1:16
            I = im2double(imread(curimages{index(ii)}));
            if length(size(I))<3
               I = repmat(I, 1, 1, 3); 
            end
            image_id = image_ids{ii}{1};
            allI(:,:, :, ii) = imresize(I, [200 200]); %insertText(imresize(I, [100 100]), [10, 10], image_id(2:end-1));
        end
        toc;

        figure();
        montage(allI);
%         montage(allI(:,:,:,1:16));
        title([tag1 ', ' tag2 ' offset' offset]);
    end
end


end


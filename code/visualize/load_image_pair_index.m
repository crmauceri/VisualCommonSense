function [ datajson ] = load_image_pair_index(root, query, tag)
%LOAD_IMAGE_PAIR_INDEX Loads a json containing all the images labeled with
%tag sorted by other tags.
%   Note that the index tag must be the tag with the lower index in the pair.

datapath = sprintf('%s\\output\\%s\\pair\\%d.txt', root, query, tag);
datastr = importdata(datapath);
datajson = JSON.parse(datastr{1});

end


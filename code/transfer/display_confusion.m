function [ h ] = display_confusion( confusion, axis_labels )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
h = figure('Position', [100, 100, 800, 800]); 
imagesc(log(confusion));
set( gca(), 'XTickLabel', axis_labels, 'XTick', 1:length(axis_labels));
set( gca(), 'YTickLabel', axis_labels, 'YTick', 1:length(axis_labels));

%Code from http://stackoverflow.com/questions/3942892/how-do-i-visualize-a-matrix-with-colors-and-values-displayed/3943939#3943939
textStrings = num2str(round(confusion(:)), '%0.0f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(confusion,2), 1:size(confusion, 1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(confusion(:) < midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

ylabel('Actual Relationship');
xlabel('Predicted Relationship');
xticklabel_rotate;

end


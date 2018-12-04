% Draw predicted boxes on an image from a heatmap
function [labeled_img, positions] = drawLabeledBoxes(img, heat, min_box_size)
    % Label each cluster on the heatmap
    [labeled_arr, num_boxes] = bwlabel(heat);
    
    positions = zeros(0,4);
    for i=1:num_boxes
        group = (labeled_arr == i);
        [rows, cols] = find(group);
        
        % Threshold based on size of prediction
        if length(rows) >= min_box_size
            min_x = min(cols);
            min_y = min(rows);
            width = max(cols) - min_x;
            height = max(rows) - min_y;
            positions = cat(1, positions, [min_x, min_y, width, height]);
        end
    end
    
    if ~isempty(positions)
        labels = 1:size(positions,1);
    else
        error('No ships found');
    end
    
    % Draw labeled rectangles on image
    labeled_img = insertObjectAnnotation(img,'rectangle',positions,labels,...
        'LineWidth',3,'Color',{'red'});
end
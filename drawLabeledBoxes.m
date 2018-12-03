function [labeled_img, positions] = drawLabeledBoxes(img, heat, min_box_size)
    [labeled_arr, num_boxes] = bwlabel(heat);
    
    positions = zeros(0,4);
    for i=1:num_boxes
        group = (labeled_arr == i);
        [rows, cols] = find(group);
        
        if length(rows) >= min_box_size
            min_x = min(cols);
            min_y = min(rows);
            width = max(cols) - min_x;
            height = max(rows) - min_y;
            positions = cat(1, positions, [min_x, min_y, width, height]);
        end
    end
    labels = 1:size(positions,1);
    
    labeled_img = insertObjectAnnotation(img,'rectangle',positions,labels,...
        'LineWidth',3,'Color',{'red'});
end
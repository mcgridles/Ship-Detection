function [thresh_map] = thresholdHeatmap(heat, threshold)
    image_size = size(heat);
    thresh_map = zeros(image_size);

    % Threshold image
    for x = 1:image_size(1)
        for y = 1:image_size(2)
            if heat(x,y) >= threshold
                thresh_map(x,y) = 1;
            end
        end
    end
end
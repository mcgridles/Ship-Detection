function [bounding_boxes] = heatmap2BBox(input_image,threshold)
% Performs Non-max suppression on a heat map image
% threshold = how many "hits" is a true detection

image_size = size(input_image);
output_image = zeros(image_size);

% Threshold image
for x = 1:image_size(1)
    for y = 1:image_size(2)
        if input_image(x,y) >= threshold
            output_image(x,y) = 1;
        end
    end
end

% Get bounding boxes for ships in the image
bounding_boxes = [];
for x = 1:image_size(1)
    for y = 1:image_size(2)
        if output_image(x,y) == 1
            
            % Check if it is part of a ship
            is_known = 0;
            ship_count = size(bounding_boxes);
            for index = 1:4:ship_count(2)
                if x >= bounding_boxes(index) && x <= bounding_boxes(index+2) ...
                        && y >= bounding_boxes(index+1) && y <= bounding_boxes(index+3)
                    % Point is part of a ship
                    is_known = 1;
                    break;
                end
            end
            
            % Point is not part of a ship
            if ~is_known
                starting_point = [x,y];
                current_point = [x,y];
                x_min = image_size(1);
                x_max = 0;
                y_min = image_size(2);
                y_max = 0;
                discovery_map = zeros(size(output_image));
                discovery_map(starting_point(1),starting_point(2)) = 1;
                
                loop_done = 0;
                while 1
                    current_point = get_next_point(current_point,output_image,image_size,discovery_map);
                    
                    % Update bounding box
                    x_min = min(current_point(1),x_min);
                    x_max = max(current_point(1),x_max);
                    y_min = min(current_point(2),y_min);
                    y_max = max(current_point(2),y_max);

                    % Check if we're back at the start
                    if discovery_map(current_point(1),current_point(2)) == 1
                        loop_done = 1;
                    else
                        discovery_map(current_point(1),current_point(2)) = 1;
                    end
                    
                    % End while loop
                    if loop_done
                        break;
                    end
                end
                
                % Add bounding box to ships
                bounding_boxes = [bounding_boxes,[x_min,y_min,x_max,y_max]];
            end
                    
            
            % Find ship bounding box
        else
            continue;
        end
    end
end
end

function [pixel] = get_next_point(current_pixel,image,im_size,map)

    % Search all pixels surrounding current pixel
    for x1 = max(current_pixel(1)-1,1):min(current_pixel(1)+1,im_size(1))
        for y1 = max(current_pixel(2)-1,1):min(current_pixel(2)+1,im_size(2))
            
            % Don't revisit pixels
            if image(x1,y1) ~= 0
                if  map(x1,y1) == 1
                    continue;
                end
                
                % Find a pixel that borders a 0 (edge)
                for x0 = max(x1-1,1):min(x1+1,im_size(1))
                    for y0 = max(y1-1,1):min(y1+1,im_size(2))
                        if image(x0,y0) == 0
                            pixel = [x1,y1];
                            return;
                        end
                    end
                end
            end
        end
    end
    
    % No solution found
    pixel = current_pixel;
end
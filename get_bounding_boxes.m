
% Load file
file = fopen('train_ship_segmentations_v2.csv');

% Read first header line (ignore it really)
header = fgetl(file);

% Read next x lines of data
% Don't try them all at once there's 231,000
for x = 1:10
    
    % Manually split lines with ',' delimeter
    line = strsplit(fgetl(file),',');
    
    % Check if there is a ship in this image
    if line(1,2) ~= ""
        
        % Get RLE info
        rle = strsplit(char(line(1,2))," ");
        rle_size = size(rle);
        
        % Load image
        image_name = char("train_v2/" + char(line(1,1)));
        image_size = size(image);
        image = imread(image_name);
        figure(1);
        imshow(image);
        
        % Create blank mask
        mask = uint8(zeros(image_size));
        
        % Bounding box initial parameters
        x_min = image_size(1);
        x_max = 0;
        y_min = image_size(2);
        y_max = 0;
        
        % Calculate mask
        for index = 1:2:rle_size(2)
            row = floor(str2num(rle{index})/image_size(1));
            col = mod(str2num(rle{index}),image_size(1));
            pixel_count = rle{index+1};
            
            % 'Write' pixels to mask
            for offset = 0:pixel_count-1
                
                % Modify bounding box if pixel is inside image boundaries
                if col+offset < image_size(2)
                    x_min = min([col+offset,x_min]);
                    x_max = max([col+offset,x_max]);
                    y_min = min([row,y_min]);
                    y_max = max([row,y_max]);
                    
                else
                    % Ignore this for now, it seems to not care about
                    % boundaries, but obviously a ship wont be split to
                    % separate halves of one image though
                end
            end
            
            % Construct bounding box
            mask(x_min:x_max,y_min:y_max,:) = 1;
        end
        
        % Detection
        detection_img = image.*mask;
        figure(2);
        imshow(detection_img);
        
        output = sprintf("Bounding box location: [%i,%i],[%i,%i]",x_min,y_min,x_max,y_max);
        disp(output);
        disp("Next");
        
    % No ship
    else
        disp("No ships in this image");
    end
end

function [] = getBoundingBoxes(params)

    % Load input file
    input_file = fopen(fullfile(params.root_dir,'train_ship_segmentations_v2.csv'));
    
    % Read first header line (ignore it really)
    header = fgetl(input_file);
    
    % Create training file
    training_file = fopen(fullfile(params.root_dir, 'train_detections.csv'),'w');
    [train_ship_counter, train_no_ship_counter] = extractBoxes(input_file, ...
        training_file, params.root_dir, params.train_image_count);
    fclose(training_file);
    
    % Create testing file
    testing_file = fopen(fullfile(params.root_dir, 'test_detections.csv'),'w');
    [test_ship_counter, test_no_ship_counter] = extractBoxes(input_file, ...
        testing_file, params.root_dir, params.test_image_count);
    fclose(testing_file);
    
    fclose(input_file);
    
    disp("Positive/Negative sample ratio:");
    fprintf("Training: %i/%i\n",train_ship_counter,train_no_ship_counter);
    fprintf("Testing: %i/%i\n",test_ship_counter,test_no_ship_counter);
end

function [ship_counter, no_ship_counter] = extractBoxes(input_file, ...
    output_file, root_dir, num_images)

    % Read next x lines of data
    % Don't try them all at once there's 231,000
    current_img = "";
    bounding_boxes = [];
    image_counter = 0;
    ship_counter = 0;
    no_ship_counter = 0;
    while image_counter <= num_images

        % Manually split lines with ',' delimeter
        line = strsplit(fgetl(input_file),',');

        % Check if there is a ship in this image
        image_name = fullfile(root_dir, "train_v2", char(line(1,1)));
        if line(1,2) ~= ""
            ship_counter = ship_counter+1;

            % Get RLE info
            rle = strsplit(char(line(1,2))," ");
            rle_size = size(rle);

            % Load image
            if ~strcmp(image_name,current_img)
                image_counter = image_counter+1;

                % Output previous bounding boxes
                if ~strcmp(current_img,"")

                    fprintf(output_file,'%s',current_img);
                    for point = bounding_boxes.'
                        fprintf(output_file,',%i,%i,%i,%i',point(1),...
                            point(2),point(3),point(4));
                    end
                    fprintf(output_file,',1\n');

                    bounding_boxes = [];
                end

                % Get new image
                image = imread(image_name);
                image_size = size(image);
                current_img = image_name;
                
%                 figure(1);
%                 imshow(image);
            end

%             % Create blank mask
%             mask = uint8(zeros(image_size));

            % Bounding box initial parameters
            x_min = image_size(1);
            x_max = 1;
            y_min = image_size(2);
            y_max = 1;

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
                x_min = max(1,x_min);
                y_min = max(1,y_min);
                %mask(x_min:x_max,y_min:y_max,:) = 1;
            end

%             % Detection
%             detection_img = image.*mask;
%             figure(2);
%             imshow(detection_img);

            bounding_boxes = [bounding_boxes; [x_min,y_min,x_max,y_max]];

        % No ship
        else
            no_ship_counter = no_ship_counter+1;

            % Get new image
            image = imread(image_name);
            image_size = size(image);

            % Generate random "false" sample
            % Uses median sample size of 650 pixels (25x26)
            sample_x = randi([1,image_size(1)-25],1,1);
            sample_y = randi([1,image_size(2)-26],1,1);
            fprintf(output_file,'%s,%i,%i,%i,%i,0\n',...
                char(root_dir + "\train_v2/" + char(line(1,1))),...
                sample_x,sample_y,...
                sample_x+25,sample_y+25);
        end
    end
end

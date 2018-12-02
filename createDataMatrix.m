function [data] = createDataMatrix(params, root_dir, image_dir)
    % Extract features on test image to determine number of features
    test_img = imread(fullfile(root_dir,'test_images/ship_example.png'));
    test_img = smoothImage(test_img, params.filter_size);
    test_features = featureExtraction(test_img, params);
    num_features = length(test_features);

    data.features = zeros(0, num_features);
    data.class = zeros(0, 1);
    data.image_name = [];

    % Read CSV file containing detections
    input_file = fopen(fullfile(root_dir, 'detections.csv'));
    
    line = fgetl(input_file);
    while ischar(line)
        line = strsplit(line,',');
        
        % Get image name and read image
        image_path = line{1};
        image_name = strsplit(image_path,{'/','\'});
        image_name = image_name{1,end};
        image = imread(fullfile(image_dir,image_name));
        
        % Get bounding box coordinates and class number
        if isempty(line{end})
            % Don't read last element if empty cell
            boxes = line(2:end-2);
            class = str2double(line{end-1});
        else
            boxes = line(2:end-1);
            class = str2double(line{end});
        end
        
        % Make sure class was read correctly
        if class ~= 0 && class ~= 1
            error('Invalid class found: %d', class);
        end

        for b=1:4:size(boxes,2)
            % Coordinates of upper left and lower right corners of bounding
            % boxes
            x1 = str2double(boxes(b));
            y1 = str2double(boxes(b+1));
            x2 = str2double(boxes(b+2));
            y2 = str2double(boxes(b+3));

            % Smooth and extract features
            img = image(y1:y2,x1:x2,:);
            img = smoothImage(img, params.filter_size);
            features = featureExtraction(img, params);

            data.features = cat(1, data.features, features);
            data.class = cat(1, data.class, class);
            data.image_name = cat(1,data.image_name, image_name);
        end
        
        % Read next line
        line = fgetl(input_file);
    end
end
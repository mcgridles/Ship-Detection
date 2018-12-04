% Calculate prediction error for an image
function [err] = calculateError(mdl, params)
    num_correct = 0;
    total_pred = 0;
    
    filename = 'test_detections.csv';
    file = fopen(fullfile(params.root_dir,filename));
    
    line = fgetl(file);
    while ischar(line)
        line = strsplit(line,',');
        
        % Load image
        image_path = line{1};
        test_image_name = strsplit(image_path,{'/','\'});
        test_image_name = test_image_name{end};
        test_image_path = fullfile(params.root_dir, params.image_dir,...
            test_image_name);
        test_image = imread(test_image_path);
        
        % Read boxes and class
        if isempty(line{end})
            % Don't read last element if empty cell
            boxes = line(2:end-2);
            class = str2double(line{end-1});
        else
            boxes = line(2:end-1);
            class = str2double(line{end});
        end
        
        % Extract box from image
        for b=1:4:size(boxes,2)
            min_x = str2double(boxes{b});
            min_y = str2double(boxes{b+1});
            max_x = str2double(boxes{b+2});
            max_y = str2double(boxes{b+3});
            
            % Extract features
            box = test_image(min_y:max_y,min_x:max_x,:);
            box = smoothImage(box, params.filter_size);
            features = featureExtraction(box, params);
            
            % Predict class
            pred_label = predict(mdl, double(features));
            
            if pred_label == class
                num_correct = num_correct + 1;
            end
            total_pred = total_pred + 1;
        end
        
        line = fgetl(file);
    end
    fclose(file);
    
    % Calculate total error
    err = 1 - (num_correct / total_pred);
end
% Calculate prediction error for an image
function [total_err] = calculateError(params)
    total_err = zeros(0,1);
    
    % Extract data from test detections file
    ground_truth = getGroundTruth(params.root_dir);
    for i=1:size(ground_truth,1)
        image_name = ground_truth{i,1};
        boxes = ground_truth{i,2};
        
        % Load each image and classify
        image = imread(fullfile(params.root_dir, 'train_v2', image_name));
        pred = classifier(mdl, image_path, params);
        heat = thresholdHeatmap(pred, params.heatmap_thresh);
        [~, pred_pos] = drawLabeledBoxes(image, heat, params.min_box_size);
        
        % Associate ground truth data with prediction data
        [gt_boxes, pred_boxes] = associateBoxes(boxes, pred_pos);

        % Calculate error between ground truth and predictions
        for j=1:length(gt_boxes)
            pred_area = pred_boxes(j,3) * pred_boxes(j,4);
            overlap_area = rectint(gt_boxes(j,:), pred_boxes(j,:));

            % How error is calculated:
            % If the predicted bounding box is fully contained within the ground
            % truth box, the error is 0, if it is fully outside the error is 1.
            %
            % Therefore, it is the fraction of the predicted box that overlaps with
            % the ground truth box.
            box_err = 1 - (overlap_area / pred_area);
            total_err = cat(1, total_err, box_err);
        end
    end
    
    total_err = mean(total_err);
end

% Extract position and size of ground truth boxes
function [gt] = getGroundTruth(root_dir)
    filename = 'test_detections.csv';
    file = fopen(fullfile(root_dir,filename));
    gt = cell(0,2);
    
    line = fgetl(file);
    while ischar(line)
        line = strsplit(line,',');
        
        gt_image_path = line{1};
        gt_image_name = strsplit(gt_image_path,{'/','\'});
        
        if isempty(line{end})
            % Don't read last element if empty cell
            boxes = line(2:end-2);
            class = str2double(line{end-1});
        else
            boxes = line(2:end-1);
            class = str2double(line{end});
        end
        
        % Only extract boxes that contain ships
        %
        % TODO: Change this so ship and non-ship images are loaded and
        % compare by class
        %
        % If class labels are different, overlap is error
        gt_boxes = zeros(0,4);
        if class == 1
            for b=1:4:size(boxes,2)
                gt_min_x = boxes(b);
                gt_min_y = boxes(b+1);
                gt_width = boxes(b+2) - gt_min_x;
                gt_height = boxes(b+3) - gt_min_y;
                gt_boxes = cat(1, gt_boxes, [gt_min_x, gt_min_y, gt_width, gt_height]);
            end
        end
        
        data = cat(2, gt_image_name(1,end), {gt_boxes});
        gt = cat(1, gt, data);
        
        line = fgetl(file);
    end
    
    fclose(file);
end

% Find boxes with closes centers
function [gt_boxes, pred_boxes] = associateBoxes(gt_pos, pred_pos)
    gt_boxes = zeros(length(gt_pos), 4);
    pred_boxes = gt_boxes;
    
    for i=1:length(gt_pos)
        gt_boxes(i,:) = gt_pos(i,:);
        
        % Find center of ground truth box
        gt_center_x = gt_pos(i,1) + ceil(gt_pos(i,3)/2);
        gt_center_y = gt_pos(i,2) + ceil(gt_pos(i,4)/2);
        gt_center = [gt_center_x, gt_center_y];
        
        min_dist = Inf;
        min_idx = 0;
        for j=1:length(pred_pos)
            % Find center of predicted box
            pred_center_x = pred_pos(i,1) + ceil(pred_pos(i,3)/2);
            pred_center_y = pred_pos(i,2) + ceil(pred_pos(i,4)/2);
            pred_center = [pred_center_x, pred_center_y];
            
            % Calculate distance between centers
            center_dist = pdist2(gt_center, pred_center);
            if center_dist < min_dist
                % Save smallest distance
                min_dist = center_dist;
                min_idx = j;
            end
        end
        
        pred_boxes(i,:) = pred_pos(min_idx,:);
    end
end
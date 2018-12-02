% Feature extraction parameters
params.filter_size = [5 5];
params.color_space = 'rgb';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_size = [32 32];
params.visualize = false;

% Other parameters
window_size = [5 5];
training_image_count = 2000;
test_image_display_count = 10;
pixel_step_size = 10;
heatmap_threshold = 7;

% Directory variables
root_dir = './';
image_dir = 'train_v2';

%% Get bounding boxes
if ~exist(fullfile(root_dir,'detections.csv'),'file')
    disp("No file \'detections.csv\' found, extracting bounding boxes...");
    tic
    get_bounding_boxes(root_dir,training_image_count);
    runtime = toc;
    fprintf("Extracted bounding boxes in: %.8f seconds\n",runtime);
else
    disp("detections.csv found, skipping bounding box extraction");
end

%% Extract data
if ~exist('data','var')
    disp("No existing feature set found, creating new one...");
    tic
    data = createDataMatrix(params, root_dir, image_dir);
    runtime = toc;
    fprintf("Created data matrix in: %.8f seconds\n",runtime);
else
    disp("Existing feature set found, skipping feature generation");
end

%% Create classifier
if ~exist('mdl','var')
    disp("No existing classifier found, creating new one...");
    tic
    disp('Training model...');
    mdl = fitcsvm(double(data.features), double(data.class),'Holdout',0.15);

    compact_mdl = mdl.Trained{1}; % Extract the trained, compact classifier
    testInds = test(mdl.Partition);   % Extract the test indices
    X_test = double(data.features(testInds,:));
    y_test = double(data.class(testInds,:));
    mdl_loss = loss(compact_mdl, X_test, y_test); % Calculate loss

    saveCompactModel(compact_mdl, fullfile(root_dir,'ship_detection_model_v2'));
    runtime = toc;
    fprintf("Generated model in: %.8f seconds\n",runtime);
else
    disp("Found existing classifier, skipping classifier creation");
end

%% Display test images
index_offset = 0;
for index = 1:test_image_display_count
    for test_index = index+index_offset:size(y_test,1)
        if y_test(test_index) == 1
            
            test_image_name = data.image_name(test_index,:);
            
            % Run image through classifier
            heatmap = classifier(root_dir,compact_mdl,image_dir,test_image_name,window_size,pixel_step_size,params); 
            
            % Display image
            test_image = imread(fullfile(root_dir,image_dir,test_image_name));
            
            % Retrieve bounding boxes
            bounding_boxes = heatmap2BBox(heatmap,heatmap_threshold);
            
            % Retrieve groundtruth bounding boxes
            file = fopen(fullfile(root_dir,'detections.csv'));
            line = fgetl(file);
            groundtruth = [];
            while ischar(line)
                line = strsplit(fgetl(file),',');
                if contains(char(line(1,1)),test_image_name)
                    class = line(1,end);
                    if class == 1
                        groundtruth = line(1,2:end-1);
                    end
                end
                line = fgetl(file);
            end
            
            % Compare with groundtruth
            figure(1);
            imshow(test_image);
            hold on;
            for bbox_index = 1:4:size(bounding_boxes,1)
                rectangle('Position',bounding_boxes(bbox_index:bbox_index+3),...
                    'EdgeColor','b');
                hold on;
            end
            for gbox_index = 1:4:size(groundtruth,1)
                rectangle('Position',[bounding_boxes(bbox_index:bbox_index+3)],...
                    'EdgeColor','g');
                hold on;
            end
            
            % Update offset
            index_offset = test_index+1;
            break;
        end
    end
end
        

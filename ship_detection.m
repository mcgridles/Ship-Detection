% Feature extraction parameters
params.filter_size = [5 5];
params.color_space = 'ycbcr';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_size = [32 32];
params.visualize = false;

% Other parameters
window_size = [30 30];
training_image_count = 75;
test_image_display_count = 10;
pixel_size = 100;
heatmap_threshold = 15;

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
    disp("Found detections.csv, skipping bounding box extraction");
end

%% Extract data
if ~exist(fullfile(root_dir, 'data.mat'),'file')
    disp("No existing feature set found, creating new one...");
    tic
    data = createDataMatrix(params, root_dir, image_dir);
    save(fullfile(root_dir,'data.mat'), 'data');
    feature_extract_runtime = toc;
    fprintf("Created data matrix in: %.8f seconds\n",feature_extract_runtime);
else
    disp("Found existing feature set, skipping feature generation");
    data_struct = load(fullfile(root_dir, 'data.mat'));
    data = data_struct.data;
end

if params.visualize
    tic
    [pcs,scrs,~,~,pexp] = pca(double(data.features));
    pca_runtime = toc;
    fprintf("Performed PCA in: %.8f seconds\n",pca_runtime);
    
    pareto(pexp);
    title('Result of PCA, first 10 components');
    
    data_reduced = srcs(:,1:3);
    scatter3(data_reduced(:,1), data_reduced(:,2), data_reduced(:,3));
    xlabel('Component 1');
    ylabel('Component 2');
    zlabel('Component 3');
    title('3 Most dominant components');
end

%% Create classifier
if ~exist(fullfile(root_dir, 'ship_detection_model.mat'), 'file')
    disp("No existing classifier found, creating new one...");
    tic
    disp('Training model...');
    mdl = fitcsvm(double(data.features), double(data.class),'HoldOut',0.15);

    compact_mdl = mdl.Trained{1}; % Extract the trained, compact classifier
    testInds = test(mdl.Partition);   % Extract the test indices
    X_test = double(data.features(testInds,:));
    y_test = double(data.class(testInds,:));
    mdl_loss = loss(compact_mdl, X_test, y_test); % Calculate loss

    saveCompactModel(compact_mdl, fullfile(root_dir,'ship_detection_model'));
    training_runtime = toc;
    fprintf("Generated model in: %.8f seconds\n",training_runtime);
    
    mdl = compact_mdl;
else
    disp("Found existing classifier, skipping classifier creation");
    mdl = loadCompactModel(fullfile(root_dir, 'ship_detection_model.mat'));
end

%% Single image classification
test_image_name = fullfile(root_dir,'test_images','00a3ab3cc.jpg');
test_image = imread(test_image_name);

disp('Making predictions...');
pred = classifier(mdl, test_image_name, params, window_size, pixel_size);

figure(1);
h = heatmap(double(pred), 'MissingDataColor', [1 1 1],'GridVisible','off');

bounding_boxes = heatmap2BBox(pred,heatmap_threshold);

% Retrieve groundtruth bounding boxes
file = fopen(fullfile(root_dir,'detections.csv'));
line = fgetl(file);
groundtruth = [];
while ischar(line)
    line = strsplit(line,',');
    if contains(char(line(1,1)),test_image_name)
        class = str2double(line(1,end));
        if class == 1
            groundtruth = line(1,2:end-1);
        end
    end
    line = fgetl(file);
end
if ~isempty(groundtruth)
    groundtruth = str2double(groundtruth);
end

% Compare with groundtruth
figure(2);
imshow(test_image);
hold on;
for bbox_index = 1:4:size(bounding_boxes,2)
    rectangle('Position',...
        [bounding_boxes(bbox_index+1),...
        bounding_boxes(bbox_index),...
        bounding_boxes(bbox_index+3)-bounding_boxes(bbox_index+1),...
        bounding_boxes(bbox_index+2)-bounding_boxes(bbox_index)],...
        'EdgeColor','b');
    hold on;
end
for gbox_index = 1:4:size(groundtruth,2)
    rectangle('Position',...
        [groundtruth(gbox_index+1),...
        groundtruth(gbox_index),...
        groundtruth(gbox_index+3)-groundtruth(gbox_index+1),...
        groundtruth(gbox_index+2)-groundtruth(gbox_index)],...
        'EdgeColor','g');
    hold on;
end
return

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
                line = strsplit(line,',');
                if contains(char(line(1,1)),test_image_name)
                    class = str2double(line(1,end));
                    if class == 1
                        groundtruth = line(1,2:end-1);
                    end
                end
                line = fgetl(file);
            end
            if ~isempty(groundtruth)
                groundtruth = str2double(groundtruth);
            end
            
            % Compare with groundtruth
            figure(3);
            imshow(test_image);
            hold on;
            for bbox_index = 1:4:size(bounding_boxes,2)
                rectangle('Position',...
                    [bounding_boxes(bbox_index+1),...
                    bounding_boxes(bbox_index),...
                    bounding_boxes(bbox_index+3)-bounding_boxes(bbox_index+1),...
                    bounding_boxes(bbox_index+2)-bounding_boxes(bbox_index)],...
                    'EdgeColor','b');
                hold on;
            end
            for gbox_index = 1:4:size(groundtruth,2)
                rectangle('Position',...
                    [groundtruth(gbox_index+1),...
                    groundtruth(gbox_index),...
                    groundtruth(gbox_index+3)-groundtruth(gbox_index+1),...
                    groundtruth(gbox_index+2)-groundtruth(gbox_index)],...
                    'EdgeColor','g');
                hold on;
            end
            hold off;
            
            % Update offset
            index_offset = test_index+1;
            break;
        end
    end
end

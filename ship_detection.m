% Feature extraction parameters
params.filter_size = [5 5];
params.color_space = 'rgb';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_size = [32 32];
params.visualize = false;
training_image_count = 2000;

% Directory variables
root_dir = './';
image_dir = 'train_v2';

%% Get bounding boxes
if ~exist(fullfile(root_dir,'detections.csv'),'file')
    disp("No file \'detections.csv\' found, extracting bounding boxes");
    tic
    get_bounding_boxes(root_dir,training_image_count);
    runtime = toc;
    fprintf("Extracted bounding boxes in: %.8f seconds\n",runtime);
else
    disp("detections.csv found, skipping bounding box extraction");
end

%% Extract data
if ~exist('data','var')
    disp("No existing feature set found, creating new one");
    tic
    data = createDataMatrix(params, root_dir, image_dir);
    runtime = toc;
    fprintf("Created data matrix in: %.8f seconds\n",runtime);
else
    disp("Existing feature set found, skipping feature generation");
end

%% Create classifier
if ~exist('mdl','var')
    disp("No existing classifier found, creating new one");
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
% Ship Detection

%% Define Parameters
% Feature extraction parameters
params.filter_size = [11 11];
params.color_space = 'ycbcr';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_size = [32 32];
params.visualize = false;

% Other parameters
params.train_image_count = 220;
params.test_image_count = 100;
params.test_image_display_count = 10;
params.window_size = [30 30];
params.num_steps = 70;
params.num_pca_features = 256;
params.pca_coeff = [];
params.confidence_thresh = 1.25;
params.heatmap_thresh = 0.3;
params.min_box_size = 20;
params.k_fold = 10;
params.poly_order = 3;

% Directory variables
params.root_dir = './Ship-Detection';
params.image_dir = 'train_v2';

%% Load data
if ~exist(fullfile(params.root_dir, 'train_detections.csv'),'file')
    disp("No file 'train_detections.csv' found, extracting bounding boxes...");
    tic
    getBoundingBoxes(params);
    bbox_runtime = toc;
    fprintf("Extracted bounding boxes in: %.8f seconds\n",bbox_runtime);
else
    disp("Found detections.csv, skipping bounding box extraction");
end

%% Extract features
if ~exist(fullfile(params.root_dir, 'data.mat'),'file')
    disp("No existing feature set found, creating new one...");
    tic
    data = createDataMatrix(params);
    save(fullfile('Ship-Detection','data.mat'), 'data');
    feature_extract_runtime = toc;
    fprintf("Created data matrix in: %.8f seconds\n",feature_extract_runtime);
else
    disp("Found existing feature set, skipping feature generation");
    data_struct = load(fullfile(params.root_dir, 'data.mat'));
    data = data_struct.data;
end

% Perform PCA on data
% PCA reduction was tested and did not produce better results
% [params.pca_coeff, data.features] = featurePCA(data, params);

%% Train classifier
if ~exist(fullfile(params.root_dir, 'ship_detection_model.mat'), 'file')
    disp("No existing classifier found, training model...");
    tic
    svm_mdl = fitcsvm(double(data.features), double(data.class), ...
        'ClassNames', [0 1], 'KernelFunction', 'polynomial', ...
        'PolynomialOrder', params.poly_order, 'CrossVal', 'on', ...
        'KFold', params.k_fold);
    
    % Find best model using KFold cross validation
    best_loss = Inf;
    for fold=1:params.k_fold
        % Extract the trained, compact classifier
        trained_mdl = svm_mdl.Trained{fold};

        test_idx = test(svm_mdl.Partition, fold); % Extract the test indices
        X_test = double(data.features(test_idx,:));
        y_test = double(data.class(test_idx,:));
        mdl_loss = loss(trained_mdl, X_test, y_test); % Calculate loss
        
        if mdl_loss < best_loss
            best_loss = mdl_loss;
            mdl = trained_mdl;
        end
    end

    saveCompactModel(mdl, fullfile(params.root_dir,'ship_detection_model'));
    training_runtime = toc;
    fprintf("Generated model in: %.8f seconds\n",training_runtime);
    
    mdl = compact_mdl;
else
    disp("Found existing classifier, skipping classifier creation");
    mdl = loadCompactModel(fullfile(params.root_dir, 'ship_detection_model.mat'));
end

% Calculate error
err = calculateError(mdl, params);
fprintf('Total error: %.3f\n', err);

%% Make predictions on single image
image_name = 'test_image.jpg';
image_path = fullfile(params.root_dir,'test_images',image_name);
image = imread(image_path);

disp('Searching image...');
pred = makePredictions(mdl, image_path, params);
heat = thresholdHeatmap(pred, params.heatmap_thresh);
[labeled_img, pos] = drawLabeledBoxes(image, heat, params.min_box_size);

figure;
heatmap(double(pred),'MissingDataColor',[1 1 1],'GridVisible','off');
figure;
heatmap(double(heat),'MissingDataColor',[1 1 1],'GridVisible','off');
figure;
imshow(labeled_img);

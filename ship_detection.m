% Feature extraction parameters
params.filter_size = [5 5];
params.color_space = 'rgb';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_size = [32 32];
params.visualize = false;

% Directory variables
root_dir = 'Ship-Detection';
image_dir = 'D:\Ship-Detection\train_v2';

tic
% Create feature matrix
disp('Extracting features...');
data = createDataMatrix(params, root_dir, image_dir);
toc

tic
% Train and save classifier
disp('Training model...');
mdl = fitcsvm(double(data.features), double(data.class));
saveCompactModel(mdl, 'Ship-Detection/ship_detection_model_v2');
toc
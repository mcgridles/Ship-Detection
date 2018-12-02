ship_img = imread('./Ship-Detection/test_images/ship_example.png');
not_ship_img = imread('./Ship-Detection/test_images/not_ship_example.png');

% Feature extraction parameters
params.filter_size = [5 5];
params.color_space = 'rgb';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_sz = [32 32];
params.visualize = false;

% Smooth images using a median filter
ship_img = smoothImage(ship_img, params.filter_size);
not_ship_img = smoothImage(not_ship_img, params.filter_size);

if params.visualize
    figure;
    imshow(ship_img, 'InitialMagnification', 'fit');
    figure;
    imshow(not_ship_img, 'InitialMagnification', 'fit');
end

% Extract features
ship_features = featureExtraction(ship_img, params);
not_ship_features = featureExtraction(not_ship_img, params);
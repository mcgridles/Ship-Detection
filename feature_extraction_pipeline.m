ship_img = imread('./Ship-Detection/test_images/ship_example.png');
not_ship_img = imread('./Ship-Detection/test_images/not_ship_example.png');

% Feature extraction parameters
filter_size = [5 5];
color_space = 'rgb';
color_bins = 32;
grad_size = [8 8];
grad_bins = 9;
spatial_sz = [32 32];
visualize = true;

% Smooth images using a median filter
ship_img = smoothImage(ship_img, filter_size);
not_ship_img = smoothImage(not_ship_img, filter_size);

if visualize
    figure;
    imshow(ship_img, 'InitialMagnification', 'fit');
    figure;
    imshow(not_ship_img, 'InitialMagnification', 'fit');
end

% Extract features
ship_features = featureExtraction(ship_img, color_space, color_bins, ...
    grad_size, grad_bins, spatial_sz, visualize);
not_ship_features = featureExtraction(not_ship_img, color_space, ...
    color_bins, grad_size, grad_bins, spatial_sz, visualize);

% Smooths each channel of an image using a median filter
function [smoothed] = smoothImage(img, filter_size)
    smoothed = img;
    for channel=1:size(img,3)
        smoothed(:,:,channel) = medfilt2(img(:,:,channel),filter_size);
    end
end
ship_img = imread('ship_example.png');
not_ship_img = imread('not_ship_example.png');
% Feature extraction parameters
filter_size = [5 5];
color_space = 'rgb';
color_bins = 32;
grad_size = [16 16];
grad_bins = 9;
spatial_sz = [32 32];
visualize = false;

% Smooth images using a median filter
ship_img = smoothImage(ship_img, filter_size);
not_ship_img = smoothImage(not_ship_img, filter_size);

% Extract features
ship_features = featureExtraction(ship_img, color_space, color_bins, ...
    grad_size, grad_bins, spatial_sz, visualize);
not_ship_features = featureExtraction(not_ship_img, color_space, ...
    color_bins, grad_size, grad_bins, spatial_sz, visualize);

%create feature matrix
featureMatrix = [double(ship_features)'; double(not_ship_features)'];
%create labels matrix
labels = [1;2];
imgname = 'test1.jpg';

%call classification function 
a = classifier(featureMatrix,labels, imgname,color_space, ...
    color_bins, grad_size, grad_bins, spatial_sz, visualize, [30 30], 200);
%create heatmap
h = heatmap(a, 'MissingDataColor', [1 1 1],'GridVisible','off');


% Smooths each channel of an image using a median filter
function [smoothed] = smoothImage(img, filter_size)
    smoothed = img;
    for channel=1:size(img,3)
        smoothed(:,:,channel) = medfilt2(img(:,:,channel),filter_size);
    end
end

close all; clear all;
%IM REALLY SORRY THIS IS SO GROSS BUT IT KINDA WORKS SO AYY

ship_img = imread('ship_example.png');
not_ship_img = imread('not_ship_example.png');

% Feature extraction parameters
filter_size = [5 5];
color_space = 'rgb';
color_bins = 32;
grad_size = [8 8];
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

%beginning classification 
example_image = imread('test1.jpg');
[rows, columns, d] = size(example_image);
img = example_image(1:32,1:32,:)
feature_1 = featureExtraction(img, color_space, color_bins,grad_size, grad_bins, spatial_sz, visualize);
featureMatrix = [double(ship_features)'; double(not_ship_features)'];
labels = [1;2];

mdl = fitcsvm(featureMatrix,labels);
predict(mdl,double(feature_1)')
s = 1;
num = 10;
for i = 1:num:(rows-spatial_sz(1))
    for j = 1:num:(columns-spatial_sz(1))
        img = example_image(j:(j+32),i:(i+32),:);
        feature = featureExtraction(img, color_space, color_bins,grad_size, grad_bins, spatial_sz, visualize);
        prediction(s) = predict(mdl, double(feature)');
        s= s+1;
    end
end
l = length( 1:num:(rows-spatial_sz(1))) ;
prediction = reshape(prediction, l,l);
heatmap(prediction)


% Smooths each channel of an image using a median filter
function [smoothed] = smoothImage(img, filter_size)
    smoothed = img;
    for channel=1:size(img,3)
        smoothed(:,:,channel) = medfilt2(img(:,:,channel),filter_size);
    end
end


 


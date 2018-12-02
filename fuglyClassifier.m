close all; clear all;
%IM REALLY SORRY THIS IS SO GROSS BUT IT KINDA WORKS SO AYY

ship_img = imread('./Ship-Detection/test_images/ship_example.png');
not_ship_img = imread('./Ship-Detection/test_images/not_ship_example.png');

% Feature extraction parameters
params.filter_size = [5 5];
params.color_space = 'rgb';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_size = [32 32];
params.visualize = false;

% Smooth images using a median filter
ship_img = smoothImage(ship_img, params.filter_size);
not_ship_img = smoothImage(not_ship_img, params.filter_size);

% Extract features
ship_features = featureExtraction(ship_img, params);
not_ship_features = featureExtraction(not_ship_img, params);

%beginning classification 
example_image = imread('./Ship-Detection/test_images/00a3ab3cc.jpg');
[rows, columns, d] = size(example_image);
img = example_image(1:32,1:32,:);
feature_1 = featureExtraction(img, params);
featureMatrix = [double(ship_features); double(not_ship_features)];
labels = [1;2];

mdl = fitcsvm(featureMatrix,labels);
predict(mdl,double(feature_1))
s = 1;
num = 10;
for i = 1:num:(rows-params.spatial_size(1))
    for j = 1:num:(columns-params.spatial_size(1))
        img = example_image(j:(j+32),i:(i+32),:);
        img = smoothImage(img, params.filter_size);
        feature = featureExtraction(img, params);
        prediction(s) = predict(mdl, double(feature));
        s= s+1;
    end
end
l = length( 1:num:(rows-params.spatial_size(1))) ;
prediction = reshape(prediction, l,l);
heatmap(prediction)


% Smooths each channel of an image using a median filter
function [smoothed] = smoothImage(img, filter_size)
    smoothed = img;
    for channel=1:size(img,3)
        smoothed(:,:,channel) = medfilt2(img(:,:,channel),filter_size);
    end
end


 


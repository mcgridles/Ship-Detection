% Smooths each channel of an image using a median filter
function [smoothed] = smoothImage(img, filter_size)
    smoothed = img;
    for channel=1:size(img,3)
        smoothed(:,:,channel) = medfilt2(img(:,:,channel),filter_size);
    end
end
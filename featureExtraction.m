% Extracts features from an RGB image
function [features] = featureExtraction(img, color_space, color_bins, ...
    grad_size, grad_bins, spatial_size, viz)

    if ~strcmp(color_space,'rgb')
        img = colorConvert(img, color_space);
    end
    
    color_features = colorHistogram(img, color_bins, viz);
    
    gray_img = rgb2gray(img);
    gradient_features = gradientHistogram(gray_img, grad_size, ...
        grad_bins, viz);
    
    spatial_features = spatialBinning(img, spatial_size, viz);
    
    features = cat(1, color_features, gradient_features, spatial_features);
end

% Turns color information into a feature vector using histograms
function [features] = colorHistogram(img, nbins, viz)
    channel_1 = img(:,:,1);
    channel_2 = img(:,:,2);
    channel_3 = img(:,:,3);

    [c1_hist,~] = histcounts(channel_1(:),nbins);
    [c2_hist,~] = histcounts(channel_2(:),nbins);
    [c3_hist,~] = histcounts(channel_3(:),nbins);
    
    features = cat(2, c1_hist, c2_hist, c3_hist);
    features = features';
    
    if viz
        figure;
        subplot(1,3,1)
        bar(c1_hist);
        title('Channel 1');
        subplot(1,3,2)
        bar(c2_hist);
        title('Channel 2');
        subplot(1,3,3)
        bar(c3_hist);
        title('Channel 3');
    end
end

% Extracts histogram of oriented gradient features from an image
function [features] = gradientHistogram(img, cell_size, nbins, viz)

    img = imresize(img, [64 64]);
    
    [features, hog_viz] = extractHOGFeatures(img, 'CellSize', cell_size, ...
        'NumBins', nbins);
    features = features';
    
    if viz
        figure;
        imshow(img, 'InitialMagnification', 'fit');
        hold on
        plot(hog_viz);
        
        title(sprintf('Oriented Gradients with cellsize=[%d, %d] and nbins=%d',...
            cell_size(1), cell_size(2), nbins));
    end
end

% Resizes and vectorizes an image
function [features] = spatialBinning(img, sz, viz)
    reduced = imresize(img, sz);
    features = reduced(:);
    
    if viz
        figure;
        plot(features);
        title(sprintf('Spatial Binning for image size=[%d, %d]', ...
            sz(1), sz(2)));
    end
end

% Converts an image from RGB to another color space
function [converted_img] = colorConvert(img, color_space)
    if strcmp(color_space,'hsv')
        converted_img = rgb2hsv(img);
    elseif strcmp(color_space, 'ycrcb')
        converted_img = rgb2ycrcb(img);
    elseif strcmp(color_space, 'lab')
        converted_img = rbg2lab(img);
    else
        converted_img = img;
    end
end
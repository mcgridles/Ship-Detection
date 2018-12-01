function [array] = classifier(featureMatrix, labels, imgName, color_space, ...
    color_bins, grad_size, grad_bins, spatial_sz, visualize, pixelSize, numberOfSteps )

%read image and extract rows and columns 
example_image = imread(imgName);
[rows, columns, d] = size(example_image);

%create support vector machine model based on featureMatrix and labels
mdl = fitcsvm(featureMatrix,labels);

%initialize heatmap array
array = zeros(rows, columns);

%loop through all the rows and columns at the specified number of steps
for i = linspace(1,rows-pixelSize(1),numberOfSteps)
    for j = linspace(1, columns-pixelSize(2),numberOfSteps)
        %round i and j to the nearest integer to use as indices
        j = round(j,0);
        i = round(i,0);
        %extract part of the image and the features from this bin
        img = example_image(i:(i+pixelSize(1)),j:(j+pixelSize(2)),:);
        feature = featureExtraction(img, color_space, color_bins,grad_size, grad_bins, spatial_sz, visualize);
        %predict if there is a boat there or not (1 means there is a boat 2
        %means there is not rn
        p = predict(mdl, double(feature)');
        %if the prediction says theres a boat increase the "score" of each
        %pixel in the bin by 1
        if p==1
            for h = i:(i+pixelSize(1))
                for g = j:(j+pixelSize(2))
                    array(h,g) = array(h,g)+1;
                end
            end
       end
    end
end

end






 


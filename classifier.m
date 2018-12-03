function [arr] = classifier(mdl, imgName, params)
    % read image and extract rows and columns 
    example_image = imread(imgName);
    [rows, columns, ~] = size(example_image);

    % initialize heatmap array
    arr = zeros(rows, columns);

    % Loop through windows of different sizes
    for w=1:size(params.window_size,1)
        win = params.window_size(w,:);
        
        % loop through all the rows and columns at the specified number of 
        % steps
        for i = linspace(1,rows-win(1),params.num_steps)
            for j = linspace(1, columns-win(2),params.num_steps)
                % round i and j to the nearest integer to use as indices
                x = round(j,0);
                y = round(i,0);
                x_win = x+win(2);
                y_win = y+win(1);
                
                % extract part of the image and the features from this bin
                img = example_image(y:y_win,x:x_win,:);
                feature = featureExtraction(img, params);
                
                % If PCA reduction is being used
                if ~isempty(params.pca_coeff)
                    feature = feature * params.pca_coeff;
                end
                
                % predict if there is a boat there or not (1 means there 
                % is a boat 0 means there is not
                [label, score] = predict(mdl, double(feature));
                
                % if the prediction says theres a boat increase the "score" 
                % of each pixel in the bin by 1
                if score(2) >= params.confidence_thresh
                    arr(y:y_win,x:x_win) = arr(y:y_win,x:x_win) + 1;
                end
            end
        end
    end
    
    % Normalize heatmap
    max_heat = max(arr(:));
    min_heat = min(arr(:));
    arr = (arr - min_heat) / max_heat;
 end
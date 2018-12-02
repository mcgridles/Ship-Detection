function [heatmap] = classifier(root_dir, model, image_folder, image_name, window_size, pixel_step_size, extr_params)
    
    % Load image
    image = imread(fullfile(root_dir,image_folder,image_name));
    heatmap = zeros(size(image));
    
    % Traverse image and classify pixels
    for x = window_size(1)+1:pixel_step_size:size(image,1)-window_size(1)
        for y = window_size(2)+1:pixel_step_size:size(image,2)-window_size(2)
            
            % Extract features in window
            feature_window = image(x-window_size(1):x+window_size(1),y-window_size(2):y+window_size(2),:);
            feature = featureExtraction(feature_window, extr_params);
            
            % Classify feature
            p = predict(model,double(feature));
            
            % Incriment heatmap if classification indicates a ship
            if p == 1
                for hx = x-window_size(1):x+window_size(1)
                    for hy = y-window_size(2):y+window_size(2)
                        heatmap(hx,hy) = heatmap(hx,hy)+1;
                    end
                end
            end
        end
    end
end
function [data] = createDataMatrix(params, root_dir, image_dir)
    test_img = imread('./Ship-Detection/test_images/ship_example.png');
    test_img = smoothImage(test_img, params.filter_size);
    test_features = featureExtraction(test_img, params);
    num_features = length(test_features);

    input_file = fopen(fullfile(root_dir, 'test.csv'));

    data.features = zeros(num_features, 0);
    data.class = zeros(1, 0);
    while true
        line = fgetl(input_file);
        if ~ischar(line)
            break
        end
        line = strsplit(line,',');
        
        image_path = line{1};
        boxes = line(2:end-1);
        class = str2double(line{end});

        image_name = strsplit(image_path,'/');
        image_name = image_name{1,end};
        image = imread(fullfile(root_dir,image_dir,image_name));

        for b=1:4:size(boxes,2)
            x1 = str2double(boxes(1,b));
            y1 = str2double(boxes(1,b+1));
            x2 = str2double(boxes(1,b+2));
            y2 = str2double(boxes(1,b+3));

            img = image(y1:y2,x1:x2,:);
            img = smoothImage(img, params.filter_size);
            features = featureExtraction(img, params);

            data.features = cat(2, data.features, features);
            data.class = cat(2, data.class, class);
        end
    end
end
% Feature extraction parameters
params.filter_size = [5 5];
params.color_space = 'rgb';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_sz = [32 32];
params.visualize = false;

test_img = imread('./Ship-Detection/test_images/ship_example.png');
test_img = smoothImage(test_img, filter_size);
test_features = featureExtraction(test_img, params);
num_features = length(test_features);

root_dir = 'Ship-Detection';
image_dir = 'training';
input_file = fopen(fullfile(root_dir, 'detections.csv'));

% Read first line
line = strsplit(fgetl(input_file),',');

data.features = zeros(num_features, 0);
data.class = zeros(1, 0);
while ischar(line)
    image_name = line(1,1);
    boxes = line(1,end-1);
    class = str2int(line(1,end));
    
    image = imread(fullfile(root_dir,image_dir,image_name));
    
    for b=1:4:size(boxes,2)
        x1 = str2int(boxes(1,b));
        y1 = str2int(boxes(1,b+1));
        x2 = str2int(boxes(1,b+2));
        y2 = str2int(boxes(1,b+3));
        
        img = image(y1:y2,x1:x2,:);
        img = smoothImage(img, filter_size);
        features = featureExtraction(img, params);
        
        data.features = cat(2, data.features, features);
        data.class = cat(2, data.class, class);
    end
    
    line = strsplit(fgetl(input_file),',');
end
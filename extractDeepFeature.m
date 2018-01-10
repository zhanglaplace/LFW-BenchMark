function evaluation_1()

clear;clc;close all;

%% caffe setttings
matCaffe = 'D:/Deeplearning/Caffe/caffe/matlab';
addpath(genpath(matCaffe));

gpu = 1;
if gpu
    gpu_id = 0;
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end
caffe.reset_all();


model   = './face_deploy.prototxt';
weights = '../face_model.caffemodel';
net     = caffe.Net(model, weights, 'test');

load imglist.mat

%% rectangle alignment

LFWfeaturesMax=cell(0);
LFWfeaturesConcat=cell(0);
LFWfeaturesAverage=cell(0);

%% compute features
for i = 1:length(imglist)
    image = imread(imglist(i).file);
    if size(image, 3) < 3
        image(:,:,2) = image(:,:,1);
        image(:,:,3) = image(:,:,1);
    end
    image = single(image);
    image = (image - 127.5)/128;
    image = permute(image, [2,1,3]);
    image = image(:,:,[3,2,1]);
    
    cropImg_(:,:,1) = flipud(image(:,:,1));
    cropImg_(:,:,2) = flipud(image(:,:,2));
    cropImg_(:,:,3) = flipud(image(:,:,3));
    
    % extract deep feature
    res = net.forward({image});
    res_ = net.forward({cropImg_});
    
    %concat
    deepfeatureConcat = [res{1}; res_{1}];
    LFWfeaturesConcat{i} = deepfeatureConcat;
    
    %max
    deepfeatureMax = bsxfun(@max,res{1},res_{1});
    LFWfeaturesMax{i} = deepfeatureMax;
    
    %average
    deepfeatureAverage = 0.5 *(res{1}+res_{1});
    LFWfeaturesAverage{i} = deepfeatureAverage;

end

% save features for evaluation
save lfwfeatures_average.mat LFWfeaturesAverage imglist;
save lfwfeatures_max.mat LFWfeaturesMax imglist;
save lfwfeatures_concat.mat LFWfeaturesConcat imglist;
end



function pairs = parseList(list, folder)
i    = 0;
fid  = fopen(list);
line = fgets(fid);
while ischar(line)
    strings = strsplit(line, '\t');
    if length(strings) == 3
        i = i + 1;
        pairs(i).fileL = fullfile(folder, strings{1}, [strings{1}, num2str(str2num(strings{2}), '_%04i.jpg')]);
        pairs(i).fileR = fullfile(folder, strings{1}, [strings{1}, num2str(str2num(strings{3}), '_%04i.jpg')]);
        pairs(i).fold  = ceil(i / 600);
        pairs(i).flag  = 1;
    elseif length(strings) == 4
        i = i + 1;
        pairs(i).fileL = fullfile(folder, strings{1}, [strings{1}, num2str(str2num(strings{2}), '_%04i.jpg')]);
        pairs(i).fileR = fullfile(folder, strings{3}, [strings{3}, num2str(str2num(strings{4}), '_%04i.jpg')]);
        pairs(i).fold  = ceil(i / 600);
        pairs(i).flag  = -1;
    end
    line = fgets(fid);
end
fclose(fid);
end

function feature = extractDeepFeature(file, net)
img     = single(imread(file));
img     = (img - 127.5)/128;
img     = permute(img, [2,1,3]);
img     = img(:,:,[3,2,1]);
res     = net.forward({img});
res_    = net.forward({flip(img, 1)});
feature = double([res{1}; res_{1}]);
end

function bestThreshold = getThreshold(scores, flags, thrNum)
accuracys  = zeros(2*thrNum+1, 1);
thresholds = (-thrNum:thrNum) / thrNum;
for i = 1:2*thrNum+1
    accuracys(i) = getAccuracy(scores, flags, thresholds(i));
end
bestThreshold = mean(thresholds(accuracys==max(accuracys)));
end

function accuracy = getAccuracy(scores, flags, threshold)
accuracy = (length(find(scores(flags==1)>threshold)) + ...
    length(find(scores(flags~=1)<threshold))) / length(scores);
end


function list = collectData(folder, name)
subFolders = struct2cell(dir(folder))';
subFolders = subFolders(3:end, 1);
files      = cell(size(subFolders));
for i = 1:length(subFolders)
    fprintf('%s --- Collecting the %dth folder (total %d) ...\n', name, i, length(subFolders));
    subList  = struct2cell(dir(fullfile(folder, subFolders{i}, '*.jpg')))';
    files{i} = fullfile(folder, subFolders{i}, subList(:, 1));
end
files      = vertcat(files{:});
dataset    = cell(size(files));
dataset(:) = {name};
list       = cell2struct([files dataset], {'file', 'dataset'}, 2);
end
clc;
clear all;
close all;



% or the features extracted by myself 
% model downloaded from "https://github.com/ydwen/caffe-face"
load('./lfwfeatures_average.mat');


feature = cell2mat(LFWfeaturesAverage);

% load pair list and label
% generated from original lfw view 2 file. 
fea_list = 'pair.label';
[label img1 img2]= textread(fea_list,'%d %s %s');

% PCA
do_pca = true;
if do_pca
    [eigvec, ~, ~, sampleMean] = PCA(feature',256);
    feature = ( bsxfun(@minus, feature', sampleMean)* eigvec )';
end

% generate scores
for i = 1:size(label,1)
    % find feature 1
    index1 = find(strcmp(struct2cell(imglist)', img1{i}) == 1);
    fea1 = feature(:,index1);
    % find feature 2
    index2 = find(strcmp(struct2cell(imglist)', img2{i}) == 1);
    fea2 = feature(:,index2);
      
    % cosine distance
    cos(i) = (fea1' * fea2)/(norm(fea1) * norm(fea2));
end

% ROC and accuracy
[fpr, tpr, auc, eer, acc] = ROCcurve(cos, label);
tmp=sprintf('ACC: %f \nEER: %f \nAUC: %f',acc,eer,auc);
disp(tmp);

plot(fpr, tpr);
axis([0,0.05,0.95,1]);
text(0.001,0.995,tmp,'EdgeColor','cyan');
legend('cos');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
hold on;


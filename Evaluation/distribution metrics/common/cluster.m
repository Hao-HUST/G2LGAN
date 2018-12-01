%% 聚类算法

% 参数说明
% feature--待聚类的feature
% clusterType--聚类类型 比如‘kmeans’
% clusterNum -- 聚类数量
function c = cluster(feature, clusterType, clusterNum)

if strcmp(clusterType, 'ap')
    dist = squareform(pdist(feature, 'euclidean'));
    S = -dist;
    p = median(S);
    c = apcluster(S, p);
    [~, ~, c] = unique(c);

elseif strcmp(clusterType, 'rcc')
    addpath(genpath('RCC'));
    filename = 'RCC/Data/temp.mat';

    %if ~exist(filename, 'file')
    data = feature;
    labels = ones(length(data),1);
    save(filename, 'data', 'labels');
    %end

    % need to run python code for finding w
    wd = cd;
    path = [wd '/RCC/Toolbox/'];
    commandStr = ['python ''', path, 'edgeConstruction.py'' --dataset feature_scene_neg.mat --samples 3558 --prep ''minmax'' --k 10 --algo ''mknn'''];
    system(commandStr);

    [c,numcomponents,optTime,gtlabels,nCluster] = RCC(filename, 100, 4);

elseif strcmp(clusterType, 'kmeans')
%     if size(feature) > 1000
%         clusterNum = 40;
%     end
%     c = kmeans(feature, clusterNum);
%     opts = statset('Display','iter','MaxIter',1000);
%     [idx,c,sumd,D] = kmeans(feature, clusterNum,'Options',opts,'Replicates',5);
    
    [c] = kmeans(feature,clusterNum,'Display','final','MaxIter',1000,'Replicates',10);

end
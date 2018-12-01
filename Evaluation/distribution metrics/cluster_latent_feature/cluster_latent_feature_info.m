%% 统计训练数据所有latent_feature的聚类情况 概率 类编号 聚类中心

% 参数说明
% path_training -- 训练数据聚类结果的路径
% partsNum -- part数量
% result -- 训练数据聚类结果的概率 类编号 聚类中心

function [probabilities,nos,centers] = cluster_latent_feature_info(path_training)
    data_training = load(path_training);
    feature_training = data_training.instance;
    probabilities = feature_training.probability;
    nos = feature_training.no;
    centers = feature_training.center;
end
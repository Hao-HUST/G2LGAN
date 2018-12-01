%% 统计训练数据所有两两连接关系的聚类情况 概率 类编号 聚类中心

% 参数说明
% path_training -- 训练数据聚类结果的路径
% partsNum -- part数量
% result -- 训练数据聚类结果的概率 类编号 聚类中心

function [ result ] = cluster_adj_pair_info(path_training,partsNum)
    for i=1:partsNum+1
       for j=i:partsNum+1
            name_sub = strcat('_',num2str(i),'_',num2str(j),'.mat');
            current_path = strrep(path_training,'.mat',name_sub);
            data_training = load(current_path);
            feature_training = data_training.instance;
            probabilities_current = feature_training.probability;
            nos_current = feature_training.no;
            centers_current = feature_training.center;
            probabilities{i,j} = probabilities_current;
            nos{i,j} = nos_current;
            centers{i,j} = centers_current;
       end
    end  
    result = {probabilities,nos,centers};
end

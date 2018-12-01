%% 统计训练数据所有part_i 与其他part连接关系的聚类情况 概率 类编号 聚类中心

% 参数说明
% path_training -- 训练数据的路径
% path_cluster_adj_parts -- 训练数据聚类结果的路径 
% clusterNum -- part数量

% 算法说明
% 当输入数据为训练数据时 根据config.clusterNums 传入的参数 使用kmeans算法将其聚成相应个数的类
% 并将聚类结果保存
% 保存数据格式为 struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
% 保存文件命名为modelName_i.mat  即part_i 与其他part连接关系的聚类结果

function cluster_adj_parts_t( path_training,path_cluster_adj_parts,clusterNum )
    data_training = load(path_training);
    adj_training = data_training.instance;
    dim = size(adj_training,2);
    spmd
    for i = 1:dim
        feature_training_cell = adj_training(i);
        feature_training = feature_training_cell{1};
        c = cluster(feature_training,'kmeans', clusterNum);
        [probabilities_training,nos_training,centers_training] = cluster_statistic(c, feature_training);
        instance = struct('probability',probabilities_training,'no',nos_training,'center',centers_training);
        save_sub = strcat('cluster/',num2str(clusterNum));
        save_path = strrep(path_cluster_adj_parts, 'cluster',save_sub);
        name_sub = strcat('_',num2str(i),'.mat');
        save_path = strrep(save_path,'.mat',name_sub);
        mysave(save_path, instance);
    end
    end

end

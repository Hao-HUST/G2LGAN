%% 聚类分析part_i与其他part之间的连接关系
% 使用kmeans方法对trainingSet part_i与其他part之间的连接关系聚类
% 然后看生成数据落在哪些类里面，从而得到在该聚类下的分布情况

% 参数说明
% path_adj_parts--待聚类的数据路径
% path_cluster_adj_parts--聚类结果保存路径
% path_cluster_adj_parts_t --训练数据聚类结果路径
% config -- 带聚类的数据的数据类型 T-训练数据 G-生成数据;聚类数目等控制参数

%算法说明
% 1、当输入数据为训练数据时 根据config.clusterNums 传入的参数 使用kmeans算法将其聚成相应个数的类
% 并将聚类结果保存，数据格式为 struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
% 2、当输入数据为生成数据时 判别生成数据落在训练数据所聚成的类中的哪一个里面，并计算其概率分布
% 结果保存数据格式为struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
% 3、按聚类统计得到的概率分布，计算其JSD
% 并保存其结果，数据格式为struct(filename,jsd_value);
% 4、按聚类情况，统计生成数据能覆盖训练数据所聚成的类中的多少个
% 并保存其结果，数据格式为struct(filename,coverage_value);

function cluster_adj_parts(path_adj_parts,path_cluster_adj_parts,path_cluster_adj_parts_t,config)
    clusterNums = config.clusterNums;
    partsNum = config.parts;
    type = config.type;
    for i=1:length(clusterNums)
        clusterNum = clusterNums(i);
        if strcmp(type,'T')
            % 1、当输入数据为训练数据时 根据config.clusterNums 传入的参数 使用kmeans算法将其聚成相应个数的类
            % 并将聚类结果保存 数据格式为 struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
            cluster_adj_parts_t(path_adj_parts,path_cluster_adj_parts,clusterNum);
        else
            name_temp = strcat('cluster/',num2str(clusterNum));
            path_cluster_training = strrep(path_cluster_adj_parts_t,'cluster',name_temp);
            cluster_t = cluster_adj_parts_info(path_cluster_training,partsNum);
            centers = cluster_t{3};
            [files, ~] = get_sub_dir(path_adj_parts); 
            if(~isempty(files))
                % 2、当输入数据为生成数据时 判别生成数据落在训练数据所聚成的类中的哪一个里面，并计算其概率分布
                % 结果保存数据格式为struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
                cluster_calculate(centers, files,clusterNum,path_cluster_adj_parts);
            end            
        end 
    end
    if strcmp(type,'G')
        % 3、按聚类统计得到的概率分布，计算其JSD
        % 并保存其结果，数据格式为struct(filename,jsd_value);
        cluster_adj_parts_JSD(path_cluster_adj_parts_t,path_cluster_adj_parts,config);
        % 4、按聚类情况，统计生成数据能覆盖训练数据所聚成的类中的多少个
        % 并保存其结果，数据格式为struct(filename,coverage_value);
        cluster_adj_parts_coverage(path_cluster_adj_parts_t,path_cluster_adj_parts,config);

    end
end


% 2、当输入数据为生成数据时 判别生成数据落在训练数据所聚成的类中的哪一个里面，并计算其概率分布
% 结果保存数据格式为struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
function cluster_calculate(centers_trainings, files,clusterNum,path_cluster_adj_parts)
    spmd
        for i_file=1:length(files)
           file_path = fullfile(files(i_file).folder,files(i_file).name);
           file = load(file_path);
           adj_generated = file.instance;
           dim = size(adj_generated,2);
           for i = 1:dim
                centers_training = centers_trainings{i};
                feature_generated_cell = adj_generated(i);
                feature_generated = feature_generated_cell{1};
                [ c ] = cluster_generated( feature_generated,centers_training);
                [probabilities,nos,centers] = cluster_statistic(c, feature_generated);
                instance = struct('probability',probabilities,'no',nos,'center',centers);
                save_sub = strcat('cluster/',num2str(clusterNum));
                save_path = strrep(path_cluster_adj_parts,'cluster',save_sub);
                name_sub = strcat('_',num2str(i),'.mat');
                save_path = strrep(save_path,'.mat',name_sub);
                mysave(save_path,instance);
           end
        end
    end    
    fprintf('%s -- cluster_adj_parts -- calculate success!\n',path_cluster_adj_parts);
end
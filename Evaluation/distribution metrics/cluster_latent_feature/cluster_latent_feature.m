%% 聚类分析autoencoder提取的latent feature
% 使用kmeans方法对trainingSet latent feature聚类
% 然后看生成数据落在哪些类里面，从而得到在该聚类下的分布情况

% 参数说明
% path_latent_feature--待聚类的数据路径
% path_cluster_latent_feature--聚类结果保存路径
% path_cluster_latent_feature_t --训练数据聚类结果路径
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

function cluster_latent_feature( path_latent_feature,path_cluster_latent_feature,path_cluster_latent_feature_t,config )
    clusterNums = config.clusterNums;
    type = config.type;
    for i=1:length(clusterNums)
        clusterNum = clusterNums(i);
        if strcmp(type,'T')
            % 1、当输入数据为训练数据时 根据config.clusterNums 传入的参数 使用kmeans算法将其聚成相应个数的类
            % 并将聚类结果保存，数据格式为 struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
            cluster_latent_feature_t(path_latent_feature,path_cluster_latent_feature,clusterNum);
        else
            clusterNum = clusterNums(i);
            name_temp = strcat('cluster/',num2str(clusterNum));
            path_cluster_training = strrep(path_cluster_latent_feature_t,'cluster',name_temp);
            [~,~,centers] = cluster_latent_feature_info(path_cluster_training);

            [files, ~] = get_sub_dir(path_latent_feature); 
            if(~isempty(files))
                % 2、当输入数据为生成数据时 判别生成数据落在训练数据所聚成的类中的哪一个里面，并计算其概率分布
                % 结果保存数据格式为struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
                cluster_calculate(centers, files,clusterNum,path_cluster_latent_feature);
            end
        end
    end
    if strcmp(type,'G')
        % 3、按聚类统计得到的概率分布，计算其JSD
        % 并保存其结果，数据格式为struct(filename,jsd_value);
        cluster_latent_feature_JSD(path_cluster_latent_feature_t,path_cluster_latent_feature,config);
        % 4、按聚类情况，统计生成数据能覆盖训练数据所聚成的类中的多少个
        % 并保存其结果，数据格式为struct(filename,coverage_value);
        cluster_latent_feature_coverage(path_cluster_latent_feature_t,path_cluster_latent_feature,config);
    end 
end


% 2、当输入数据为生成数据时 判别生成数据落在训练数据所聚成的类中的哪一个里面，并计算其概率分布
% 结果保存数据格式为struct(probabilities,nos,centers); 即各个类的概率 编号 聚类中心
function cluster_calculate(centers_training,files,clusterNum,path_cluster_latent_feature)
    for i=1:length(files)
       file_path = fullfile(files(i).folder,files(i).name);
       file = load(file_path);
       feature_generated = file.instance;
       [ c ] = cluster_generated( feature_generated,centers_training );
       [probabilities,nos,centers] = cluster_statistic(c, feature_generated);
       instance = struct('probability',probabilities,'no',nos,'center',centers);
       save_sub = strcat('cluster/',num2str(clusterNum));
       save_path = strrep(path_cluster_latent_feature,'cluster',save_sub);
       mysave(save_path,instance);
    end
    fprintf('%s -- cluster_latent_feature -- calculate success!\n',path_cluster_latent_feature);
end

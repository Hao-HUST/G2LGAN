%% 按聚类情况，统计生成数据能覆盖训练数据所聚成的类中的多少个

% 参数说明
% path_t--训练数据聚类结果的路径
% path_g--生成数据聚类结果的路径
% config -- 带聚类的数据的数据类型 T-训练数据 G-生成数据;聚类数目等控制参数

%算法说明

%按聚类情况，统计生成数据能覆盖训练数据所聚成的类中的多少个
% 并保存其结果，数据格式为struct(filename,coverage_value);

function cluster_latent_feature_coverage(path_t,path_g,config)
    clusterNums = config.clusterNums;
    for i=1:length(clusterNums)
        clusterNum = clusterNums(i);
        name_temp = strcat('cluster/',num2str(clusterNum));
        path_t_current = strrep(path_t,'cluster',name_temp);
        [probabilities,nos,~] = cluster_latent_feature_info(path_t_current);
        obj_t = struct('probability',probabilities,'no',nos);
        name_temp = strcat('cluster/',num2str(clusterNum));
        path_g_current = strrep(path_g,'cluster',name_temp);
        indfir=max(strfind(path_g_current,'/')); 
        path_g_current = path_g_current(1:indfir);
        [files, ~] = get_sub_dir(path_g_current); 
        if(~isempty(files))
           get_coverage_current_dir(obj_t,files);
        end   
    end
end

function  get_coverage_current_dir(obj1,files)
    instance = struct('file','','coverage',0);
    file_path = fullfile(files.folder,files.name);
    file = load(file_path);
    obj = file.instance;
    instance.file = files.name;
    instance.coverage = size(obj.no)/size(obj1.no);
    save_path = file_path;
    save_path = strrep(save_path,'cluster','cluster/coverage');
    mysave(save_path,instance);
    fprintf('%s -- cluster_latent_feature_coverage -- calculate success!\n',save_path);
end





%% 按聚类统计得到的概率分布，计算其JSD

% 参数说明
% path_t--训练数据聚类结果的路径
% path_g--生成数据聚类结果的路径
% config -- 带聚类的数据的数据类型 T-训练数据 G-生成数据;聚类数目等控制参数

% 算法说明

% 按聚类统计得到的概率分布，计算其JSD
% 并保存其结果，数据格式为struct(filename,jsd_value);
% 保存文件命名为modelName_i.mat  即part_i 与其他part连接关系的JSD结果
function cluster_adj_parts_JSD(path_t,path_g,config)
    clusterNums = config.clusterNums;
    for i=1:length(clusterNums)
        clusterNum = clusterNums(i);
        partsNum = config.parts;
        name_temp = strcat('cluster/',num2str(clusterNum));
        path_t_current = strrep(path_t,'cluster',name_temp);
        cluster_t = cluster_adj_parts_info(path_t_current,partsNum);
        probabilities = cluster_t{1};
        nos = cluster_t{2};
        obj_t = struct('probability',probabilities,'no',nos);
        name_temp = strcat('cluster/',num2str(clusterNum));
        path_g_current = strrep(path_g,'cluster',name_temp);
        indfir=max(strfind(path_g_current,'/')); 
        path_g_current = path_g_current(1:indfir);
        [files, ~] = get_sub_dir(path_g_current); 
        if(~isempty(files))
           get_JSD_current_dir(obj_t,files,partsNum);
        end   
    end
end

function  get_JSD_current_dir(obj1,files,partsNum)
    for i = 1:partsNum+1
       obj1_current = obj1(i);
       p = obj1_current.probability';
       nos_p = obj1_current.no';
       pattern = strcat('_', num2str(i),'.mat');
       files_name = struct2cell(files);
       files_name = files_name(1,:);
       files_name = cellstr(files_name);
       TF = contains(files_name,pattern);
       files_current = files(TF);
       instance = struct('file','','JSD',0);
       file_name = files_current.name;
       file_path = fullfile(files_current.folder,file_name);
       data = load(file_path);
       obj = data.instance;

       q = obj.probability';
       nos_q = obj.no';
       [c] = unique([nos_p,nos_q]);
       Q = zeros(1,size(c,2));
       for i_q = 1:size(nos_q,2)
           i_Q = find(c==nos_q(i_q)) ;
           Q(1,i_Q) = q(i_q);
       end
       P = zeros(1,size(c,2));
       for i_p = 1:size(nos_p,2)
           i_P = find(c==nos_p(i_p)) ;
           P(1,i_P) = p(i_p);
       end
       dist = JSD(P,Q);
       instance.file = files_current.name;
       instance.JSD = dist;
       save_path = file_path;
       save_path = strrep(save_path,'cluster','cluster/JSD');
       mysave(save_path,instance);
    end
    save_path = files(1).folder;
    fprintf('%s -- cluster_adj_parts_JSD -- calculate success!\n',save_path);
end
    




function plot_cluster_coverage_adj_parts( config )
%PLOT_CLUSTER_JSD_ADJ_PARTS 此处显示有关此函数的摘要
%   此处显示详细说明
    figDir = strcat('../curves/',config.category,'_',config.dim,'/');
    figDir = char(figDir);
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end
    data = {};  
    title = {};
    data_samples = {};
    names = config.names;
    paths = config.paths;
    types = config.types;
    partsNum = config.parts;
    part_names = config.part_names;
    clusterNums = config.clusterNums;
    for i=1:partsNum+1
        data_sample = [];  
        i_g = 1;
        i_gt = 1;
        for idx=1:length(names)
            name = names(idx);  
            type = types(idx);
            data_i = coverage_i_cluster(paths(idx),clusterNums,i);
            if strcmp(type,'G')
                data{i,i_g,1} = data_i;
                data{i,i_g,2} = name;
                i_g = i_g + 1;
            else
                data_sample(i_gt,:) = data_i(:,2)';
                i_gt = i_gt + 1;
            end
        end
        data_samples{i} = data_sample; 
        title{i} = strcat('coverage--(',part_names(i),')');
    end
    figName = [figDir, 'adj-cluster_parts_coverage'];
    view_subplot(data,data_samples,title,clusterNums, figName);
end

function matrix = coverage_i_cluster(path,clusterNums,i)
    pattern = strcat('_',num2str(i),'.mat');
    path = strrep(path,'.mat',pattern);
    matrix = zeros(1,2);
    for no=1:length(clusterNums)
        path_sub = strcat('coverage/',num2str(clusterNums(no)));
        current_path = strrep(path,'coverage',path_sub);
        data = load(current_path);
        obj = data.instance;
        matrix(no,1) = clusterNums(no);
        matrix(no,2) = obj.coverage;
    end
end




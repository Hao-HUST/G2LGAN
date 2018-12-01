function plot_cluster_JSD_component( config )
%PLOT_CLUSTER_JSD_COMPONENT 此处显示有关此函数的摘要
%   此处显示详细说明
    figDir = strcat('../curves/',config.category,'_',config.dim,'/');
    figDir = char(figDir);
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end
    data = {};  
    title = {};
    data_samples = {};
    shape_types = {};
    names = config.names;
    paths = config.paths;
    types = config.types;
    partsNum = config.parts;
    part_names = config.part_names;
    clusterNums = config.clusterNums;

    data_samples = [];  
    i_g = 1;
    i_gt = 1;
    for idx=1:length(names)
        name = names(idx);  
        type = types(idx);
        data_i = coverage_cluster(paths(idx),clusterNums);
        if strcmp(type,'G')
            data{i_g,1} = data_i;
            data{i_g,2} = name;
            i_g = i_g + 1;
        else
            data_samples(i_gt,:) = data_i(:,2)';
            i_gt = i_gt + 1;
        end
    end
    title = strcat('JSD');

    figName = [figDir, 'component_cluster_variation'];
    view_singleplot(data,data_samples,title,clusterNums, figName);
end

function matrix = coverage_cluster(path,clusterNums)
    matrix = zeros(1,2);
    for no=1:length(clusterNums)
        path_sub = strcat('JSD/',num2str(clusterNums(no)));
        current_path = strrep(path,'JSD',path_sub);
        data = load(current_path);
        obj = data.instance;
        matrix(no,1) = clusterNums(no);
        matrix(no,2) = obj.JSD;
    end
end

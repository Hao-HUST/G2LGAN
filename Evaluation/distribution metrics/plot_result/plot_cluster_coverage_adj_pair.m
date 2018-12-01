
function plot_cluster_coverage_adj_pair( config )
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
    for i=1:partsNum+1
        for j=1:partsNum+1 
            data_sample = [];  
            i_g = 1;
            i_gt = 1;
            for idx=1:length(names)
                name = names(idx);  
                type = types(idx);
                data_ij = coverage_ij_cluster(paths(idx),clusterNums,i,j);
                if strcmp(type,'G')
                    data{j,i_g,1} = data_ij;
                    data{j,i_g,2} = name;
                    i_g = i_g + 1;
                else
                    data_sample(i_gt,:) = data_ij(:,2);
                    i_gt = i_gt + 1;
                end
            end
            data_samples{j} = data_sample; 
            title{j} = strcat('coverage--(',part_names(i),'-',part_names(j),')');
        end
        figName = [figDir, 'adj-cluster_pair_coverage_', char(part_names(i))];
        view_subplot(data,data_samples,title,clusterNums,figName);
    end
end

function matrix = coverage_ij_cluster(path,clusterNums,i,j)
    if i>j 
        temp = i;
        i = j;
        j = temp;
    end
    pattern = strcat('_',num2str(i),'_',num2str(j),'.mat');
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

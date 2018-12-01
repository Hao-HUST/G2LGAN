function plot_GMM_adj_parts( config )
%PLOT_GMM_ADJ_PARTS 此处显示有关此函数的摘要
%   此处显示详细说明
    figDir = strcat('../curves/',config.category,'_',config.dim,'/');
    figDir = char(figDir);
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end
    data = {};  
    title = {};
    data_samples = {};
    top_step = config.top_step;
    paths = config.paths;
    types = config.types;
    names = config.names;
    partsNum = config.parts;
    part_names = config.part_names;
    shapes = config.shapes
    for x=1:partsNum+1
       i_g = 1;
       i_gt = 1;
       data_sample = []; 
       for idx=1:length(names)
           name = names(idx);  
           type = types(idx); 
           score_f = load(paths(idx));
           scores = score_f.instance;
           data_i = top_average_score(scores{x},top_step);
           if strcmp(type,'G') 
              data{x,i_g,1} = data_i;
              data{x,i_g,2} = name; 
              i_g = i_g + 1;
           else
              data_sample(i_gt,:) = data_i(:,2)';
              shape_types{i,i_g} = shapes(2*idx-1:2*idx);
              i_gt = i_gt + 1;
           end
       end
       data_samples{x} = data_sample; 
       title{x} = strcat('average-score-adjacent--(',part_names(x),')');
    end
    figName = [figDir, 'average-score-adjacent'];
    view_subplot(data,data_samples,title,top_step, figName, shape_types);
    
end

function score_averages = top_average_score(scores,step)
    [score_sorted,indices]= sort(scores,'descend');
    score_averages = [];
    for i=1:size(step,2)
        idx_end = floor(step(i)*size(scores,1)/100);
        score_current = score_sorted(1:idx_end);
        score_average = sum(score_current,1)/size(score_current,1);
        score_averages(i,1) = step(i);
        score_averages(i,2) = score_average;
    end
end

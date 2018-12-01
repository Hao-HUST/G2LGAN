function plot_GMM_component( config )
%PLOT_GMM_COMPONENT 此处显示有关此函数的摘要
%   此处显示详细说明
    figDir = strcat('../curves/',config.category,'_',config.dim,'/');
    figDir = char(figDir);
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end
    data = {};  
    data_samples = [];  
    top_step = config.top_step;
    paths = config.paths;
    types = config.types;
    names = config.names;
    
    i_g = 1;
    i_gt = 1;
    for idx=1:length(names)
       name = names(idx);  
       type = types(idx); 
       score_f = load(paths(idx));
       scores = score_f.instance;
       data_i = top_average_score(scores{1,2},top_step);
       if strcmp(type,'G') 
          data{i_g,1} = data_i;
          data{i_g,2} = name; 
          i_g = i_g + 1;
       else
          data_samples(i_gt,:) = data_i(:,2);
          i_gt = i_gt + 1;
       end
    end
    title = strcat('average-score-top-x-component');
    figName = [figDir, 'average-score-top-x-component'];
    view_singleplot(data,data_samples,title,top_step,figName); 
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



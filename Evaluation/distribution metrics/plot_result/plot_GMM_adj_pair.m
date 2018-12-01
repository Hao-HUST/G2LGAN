function plot_GMM_adj_pair( config )
%PLOT_GMM_ADJ_PAIR 此处显示有关此函数的摘要
%   此处显示详细说明
    figDir = strcat('../curves/',config.category,'_',config.dim,'/');
    figDir = char(figDir);
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end
    data = {};  
    title = {};
    top_step = config.top_step;
    paths = config.paths;
    types = config.types;
    names = config.names;
    partsNum = config.parts;
    part_names = config.part_names;
    for x=1:partsNum+1
       data_samples = {}; 
       for y=1:partsNum+1
           i_g = 1;
           i_gt = 1;
           for idx=1:length(names)
               name = names(idx);  
               type = types(idx);  
               score_f = load(paths(idx));
               scores = score_f.instance;
               if x>y
                   score_xy = scores{y,x};
               else 
                   score_xy = scores{x,y};
               end
               data_ij = top_average_score(score_xy,top_step);
               if strcmp(type,'G') 
                    data{y,i_g,1} = data_ij;
                    data{y,i_g,2} = name; 
                    i_g = i_g + 1;
               else
                   data_sample(i_gt,:) = data_ij(:,2);
                   i_gt = i_gt + 1;
               end
           end 
           data_samples{y} = data_sample';
           title{y} = strcat('average-score--(',part_names(x),'-',part_names(y),')');
       end
       figName = [figDir, 'average-score-top-x-adjacent-pair', char(part_names(x))];
       view_subplot(data,data_samples,title,top_step,figName);
    end
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


%获取聚类数据的 概率 类编号 聚类中心
function [ probabilities,nos,centers ] = cluster_statistic( c, data )
    nos = unique(c);
    counts =zeros(size(nos));
    for i=1:length(nos)   
        counts(i)=length(find(c==nos(i)));
    end
    probabilities = counts/length(c);
    centers = zeros(length(nos),length(data(1,:)));
    for i = 1:length(nos)
        no = nos(i);
        idices_data = find(c==no);
        current_cluster = data(idices_data,:);
        centers(i,:) = sum(current_cluster,1)/length(idices_data);
    end
end




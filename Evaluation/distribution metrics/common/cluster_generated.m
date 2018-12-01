% 计算生成数据落在哪个类中
function [ c ] = cluster_generated( feature_generated,training_cluster_centers )
    c = zeros(length(feature_generated(:,1)),1); 
    for i =1:length(feature_generated(:,1))
       c(i) = cluster_newfeature(feature_generated(i,:),training_cluster_centers); 
    end
end

function [ c ] = cluster_newfeature( feature,cluster_centers )
    feature = single(feature);
    cluster_centers = single(cluster_centers);
    min_d = pdist2(feature,cluster_centers(1,:),'euclidean');
    c = 1;
    for i=1:length(cluster_centers(:,1))
        d = pdist2(feature,cluster_centers(i,:),'euclidean');
        if min_d>d
           min_d = d;
           c = i;
        end
    end
end

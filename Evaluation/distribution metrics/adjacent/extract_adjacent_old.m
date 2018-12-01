%% 提取part之间的连接关系


% 参数说明
% path_model--要计算的一组model所在的文件夹路径
% path_adj_single--对每个model统计连接关系并分别保存在此目录下
% path_adj_pair--统计本次计算的一组model的两两part的连接关系，并保存在此目录下
% path_adj_parts -- 统计本次计算的一组model的每个part于其余所有part的连接关系，并保存在此目录下
% config -- 数据维度、part数量等


% 算法说明
% 1、adj_nabourhood_statistic_single
% 计算每个model中每个voxel与周围的26个voxel的关系，
% 返回一个结构体struct('count',count_matrix,'adj_matrix',matrix)
% 其中count_matrix为一维向量，表示各个part的voxel的数量
% 其中matrix为五维向量，表示连接关系，eg,(:,:,:,1,2)表示part1-part2连接关系的三维矩阵，
% (:,:,:,1,2)三维矩阵中的值为对应位置的voxel_part1--vixel_part2关系的数量


% 2、adj_nabourhood_statistic_all
% 计算一组数据中的两两连接关系及每个part于其余所有part的连接关系
% 
% 2、1计算一组数据中的两两连接关系
% eg part1-part2
% 统计每个model中，所有属于part1的voxel中，其周围有k(0<=k<=26)个属于part2的voxel的数量，k从0取值到26，
% 即得到一个长度为27的1维的向量，并除以p1的voxel数量进行归一化，表示该model的p1-p2的连接关系，其他part之间的关系同理
% 统计一组model的两两连接关系，并保存
% 保存数据格式 instance = matrix(count_part,count_part)
% matrix(i,j) = matrix(count_model,27)
% 
% 2、2计算一组数据中的任一part与其他所有part的连接关系
% eg 共有4个part p1 p2 p3 p4 现统计p1与其他所有part之间的关系
% 统计每个model中，所有属于p1的voxel中，其周围出现的 p2 p3 p4的voxel的数量，并除以p1的voxel的数量归一化，
% 得到一个长度为5的1维向量，表示p1与其他所有part之间的关系，其他part之间的关系同理
% 统计一组model的连接关系，并保存
% 保存数据格式 instance = matrix(count_part)
% matrix(i) = matrix(count_model,count_part)  
% 
function extract_adjacent( path_model,path_adj_single,path_adj_pair,path_adj_parts,config)
   [files, ~] = get_sub_dir(path_model); 
   if(~isempty(files))
       parts = config.parts;
       dims = config.dims;
       adj_nabourhood_statistic_single(files,parts,dims,path_adj_single);
       adj_nabourhood_statistic_all(path_adj_single,files,parts,path_adj_pair,path_adj_parts);
   end
end


% 1、adj_nabourhood_statistic_single
% 计算每个model中每个voxel与周围的26个voxel的关系，
% 返回一个结构体struct('count',count_matrix,'adj_matrix',matrix)
% 其中count_matrix为一维向量，表示各个part的voxel的数量
% 其中matrix为五维向量，表示连接关系，eg,(:,:,:,1,2)表示part1-part2连接关系的三维矩阵，
% (:,:,:,1,2)三维矩阵中的值为对应位置的voxel_part1--vixel_part2关系的数量
function adj_nabourhood_statistic_single(files,parts,dims,path_adj_single)
%  spmd
    for i=1:length(files)
       file_path = fullfile(files(i).folder,files(i).name);
       file = load(file_path);
       obj = file.instance;
       matrix = adj_nabourhood_single(obj, parts, dims);
       count_matrix = zeros(1,parts+1);
       for ii=1:parts+1
           idx = find(obj==ii-1);
           count_matrix(1,ii) = size(idx,1);
       end
       instance= struct('count',count_matrix,'adj_matrix',matrix);
       save_path = strcat(path_adj_single,files(i).name);
       mysave(save_path,instance);
    end
%  end    
    fprintf('adj_nbr_single -- calculate success!\n');
end


% 2、adj_nabourhood_statistic_all
% 计算一组数据中的两两连接关系及每个part于其余所有part的连接关系
% 
% 2、1计算一组数据中的两两连接关系
% eg part1-part2
% 统计每个model中，所有属于part1的voxel中，其周围有k(0<=k<=26)个属于part2的voxel的数量，k从0取值到26，
% 即得到一个长度为27的1维的向量，并除以p1的voxel数量进行归一化，表示该model的p1-p2的连接关系，其他part之间的关系同理
% 统计一组model的两两连接关系，并保存
% 保存数据格式 instance = matrix(count_part,count_part)
% matrix(i,j) = matrix(count_model,27)
% 
% 2、2计算一组数据中的任一part与其他所有part的连接关系
% eg 共有4个part p1 p2 p3 p4 现统计p1与其他所有part之间的关系
% 统计每个model中，所有属于p1的voxel中，其周围出现的 p2 p3 p4的voxel的数量，并除以p1的voxel的数量归一化，
% 得到一个长度为5的1维向量，表示p1与其他所有part之间的关系，其他part之间的关系同理
% 统计一组model的连接关系，并保存
% 保存数据格式 instance = matrix(count_part)
% matrix(i) = matrix(count_model,count_part)  
% 
function adj_nabourhood_statistic_all(path_adj_single,files,parts,path_adj_pair,path_adj_parts)
    file_path = fullfile(path_adj_single,files(1).name);
    file = load(file_path);
    data = file.instance;
    adj_single = data.adj_matrix;
    count = data.count;
    for i=1:parts+1
       for j=1:parts+1
        adj_statistic_pairs{i,j} = []; 
        adj_statistic_parts{i} = []; 
       end
    end  
    for i_file=1:length(files)
        file_path = fullfile(path_adj_single,files(i_file).name);
        file = load(file_path);
        data = file.instance;
        adj_single = data.adj_matrix;
        count = data.count;
        for i=1:parts+1
            if count(i)>0
                for j=1:parts+1
                    if i~=j
                        matrix = adj_single(:,:,:,i,j);
                        adj_matrix = zeros(1,27);
                        for ii = 1:27
                            idx = find(matrix==ii-1);
                            if ~isempty(idx)
                                adj_matrix(1,ii) = size(idx,1)/count(i); 
                            end
                        end
                        matrix_temp = adj_statistic_pairs{i,j};
                        matrix_temp(size(matrix_temp,1)+1,:) = adj_matrix(1,:);
                        adj_statistic_pairs{i,j} = matrix_temp;
                    end
                end  
            end
        end
        for i=1:parts+1
            if count(i)>0
                matrix = adj_single(:,:,:,i,i);
                adj_matrix = zeros(1,27);
                for ii = 1:27
                    idx = find(matrix==ii-1);
                    if ~isempty(idx)
                        adj_matrix(1,ii) = size(idx,1)/count(i); 
                    end
                end
                matrix_temp = adj_statistic_pairs{i,i};
                matrix_temp(size(matrix_temp,1)+1,:) = adj_matrix(1,:);
                adj_statistic_pairs{i,i} = matrix_temp;
            end
        end
        for i=1:parts+1
            if count(i)>0
                vector_pi = zeros(1,5);
                for j=1:parts+1
                    matrix = adj_single(:,:,:,i,j);
                    S = sum(matrix(:),'omitnan');
                    vector_pi(j) = S/count(i); 
                end
                vector_temp = adj_statistic_parts{i};
                vector_temp(size(vector_temp,1)+1,:) = vector_pi;
                adj_statistic_parts{i} = vector_temp;  
            end
        end
    end
    
    instance = adj_statistic_pairs;
    mysave(path_adj_pair,instance);
    fprintf('% s -- path_adj_pair -- calculate success!\n',path_adj_pair);

    instance = adj_statistic_parts;
    mysave(path_adj_parts,instance);
    fprintf('% s -- adj_nbr_statistic--parts -- calculate success!\n',path_adj_parts);
end
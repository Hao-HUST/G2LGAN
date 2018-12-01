%% 提取每个part包含的联通区域数

% 参数说明
% path_model--要计算的一组model所在的文件夹路径
% path_component--统计结果保存在此目录下
% config -- 数据维度、part数量等

% 算法说明
% 计算每个model的整体及每个part包含的连通区域数量及连通块的大小
% 统计一组数据并保存
% 保存数据格式instance = matrix(count_model,count_part+2)
% matrix(i,:) = {‘文件名’，{global联通区域数量，大小}，{part1联通区域数量，大小}，...}
function extract_component( path_model,path_component,config)
    [files, ~] = get_sub_dir(path_model); 
    if(~isempty(files))
        get_cc_current_dir(files,config.parts,path_component)
    end
end

function get_cc_current_dir(files,parts,path_component)
    instance = {};
    for i=1:length(files)
       file_path = fullfile(files(i).folder,files(i).name);
       file = load(file_path);
       obj = file.instance;
       cc = bwconncomp(obj);
       PixelIdxList = cellfun(@numel,cc.PixelIdxList);
       NumObjects = cc.NumObjects;
       instance{i,1} = files(i).name;
       component{1,1} = NumObjects;
       component{2,1} = PixelIdxList;
       instance{i,2} = component;
       for ii=1:parts
           current_obj = zeros(size(obj,1),size(obj,2),size(obj,3));
           current_obj(obj==ii) = 1;
           cc = bwconncomp(current_obj);
           PixelIdxList = cellfun(@numel,cc.PixelIdxList);
           NumObjects = cc.NumObjects;
           component{1,1} = NumObjects;
           component{2,1} = PixelIdxList;
           instance{i,ii+2} = component;
       end
       
    end
    mysave(path_component,instance);
    fprintf('%s -- component/ -- calculate success!\n',path_component); 
end

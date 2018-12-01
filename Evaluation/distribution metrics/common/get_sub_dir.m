function [files, dirs] = get_sub_dir(source_common_path, id)
    source_common_path = char(source_common_path);
    source_path = source_common_path;
    if exist('id','var')
        id = char(id);
        source_path = fullfile(source_common_path,id,'/');
    end
    dir_sub = dir(source_path);   
    [idxs] = arrayfun(@(x) ~isequal(x.name,'.')&&~isequal(x.name,'..'), dir_sub);
    dir_sub = dir_sub(idxs);
    [idxs] = arrayfun(@(x) isequal(x.isdir,1), dir_sub);
    dirs = dir_sub(idxs);
    files = dir_sub(~idxs);
end


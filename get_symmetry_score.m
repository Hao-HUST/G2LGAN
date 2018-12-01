function get_symmetry_score( path_source ,path_save,type)
%   get_symmetry_score 
%   Symmetry axis: between[0,dim/2] and [1+dim/2,dim] (dim = 32)
%   calculate the percentage (count_symmetry_voxel)/(count_all_model)
     dim = 32;
     dinfo = dir(strcat(path_source,'*.mat'));
     instance = struct('file','','symmetry',0);
     for i_file=1:length(dinfo)
        currentFile = dinfo(i_file).name;
        currentData = load(strcat(path_source,currentFile));
        instance_source = currentData.instance;
        instance_arm = int8(zeros(dim,dim,dim));
        if type == 0
            idx = instance_source > 0;
        else
            idx = instance_source == type;
        end
        instance_arm(idx) = 1;
     
        [ix,iy,iz]=ind2sub(size(instance_arm),find(instance_arm==1));
        if length(ix) <1
            instance(i_file).file = currentFile;
            instance(i_file).symmetry = 0;
            continue;
        end
        for ii =1:length(ix)
            if(iz(ii)<1+dim/2)
               if instance_arm(ix(ii),iy(ii),dim-iz(ii)+1) == 1
                   instance_arm(ix(ii),iy(ii),dim-iz(ii)+1) = 2;
                   instance_arm(ix(ii),iy(ii),iz(ii)) = 2;
               end
            end
        end
        idx=instance_arm==1;
        count_n_symmetry=sum(idx(:));
        idx=instance_arm==2;
        count_symmetry=sum(idx(:));
        instance(i_file).file = currentFile;
        instance(i_file).symmetry = count_symmetry/(count_n_symmetry+count_symmetry);
        
     end
     save(path_save,'instance');   
 
end



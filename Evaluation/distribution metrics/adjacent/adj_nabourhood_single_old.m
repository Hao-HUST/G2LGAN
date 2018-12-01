function [ matrix ] = adj_nabourhood_single( obj, parts, dim)
%	obj     model for calculate
%   parts   parts count of each model
%   dim     维度
%   matrix  返回矩阵5维 表示连接关系，eg,(:,:,:,1,2)表示part1-part2连接关系的三维矩阵，
%           (:,:,:,1,2)三维矩阵中的值为对应位置的voxel_part1--vixel_part2关系的数量
    parts = parts+1;
    obj_temp = zeros(dim,dim,dim);
    for part = 1:parts
        obj_temp(obj==part-1) = part^3;
    end
    conv = zeros(dim,dim,dim,26);
    kernel0 = zeros(3,3,3);
    kernel0(2,2,2) = -1;
    k=0;
    for i=1:3
       for j=1:3
            kernel = kernel0;
            kernel(i,j,3) = 1;
            k=k+1;
            
            conv(:,:,:,k) = convn(obj_temp,kernel,'same'); 
       end
    end
    for i=1:3
       for j=1:3
            kernel = kernel0;
            kernel(i,j,1) = 1;
            k=k+1;
            conv(:,:,:,k) = convn(obj_temp,kernel,'same'); 
       end
    end

    for j=1:3
        kernel = kernel0;
        kernel(1,j,2) = 1; 
        k=k+1;
        conv(:,:,:,k) = convn(obj_temp,kernel,'same'); 
    end
     for j=1:3
        kernel = kernel0;
        kernel(3,j,2) = 1; 
        k=k+1;
        conv(:,:,:,k) = convn(obj_temp,kernel,'same'); 
    end

    kernel = kernel0;
    kernel(2,1,2) = 1; 
    k=k+1;
    conv(:,:,:,k) = convn(obj_temp,kernel,'same'); 
    
    kernel = kernel0;
    kernel(2,3,2) = 1; 
    k=k+1;
    conv(:,:,:,k) = convn(obj_temp,kernel,'same'); 
%     count = zeros(parts+1,parts+1);
    matrix = zeros(dim,dim,dim,parts,parts);
    
    for x=1:parts
       idx = obj_temp == x^3;
       for y =1:parts
           if x~=y
               value = y^3 - x^3;
               boundary_current = zeros(dim,dim,dim);
               for i=1:k
                   conv_current = conv(:,:,:,i);
                   conv_current_x = zeros(dim,dim,dim);
                   conv_current_x(idx) = conv_current(idx);
                   idx2=conv_current_x==value; 
                   boundary_current(idx2) =  boundary_current(idx2)+1; 
               end
               matrix(:,:,:,x,y) = boundary_current;
           end     
       end
    end
    for x=1:parts
        idx = obj_temp == x^3;
        boundary_current = zeros(dim,dim,dim);
        for i=1:k
           conv_current = conv(:,:,:,i);
           conv_current_x = ones(dim,dim,dim);
           conv_current_x(idx) = conv_current(idx);
           idx2=conv_current_x==0;
           boundary_current(idx2) =  boundary_current(idx2)+1; 
       end
       matrix(:,:,:,x,x) = boundary_current;
    end
    
end


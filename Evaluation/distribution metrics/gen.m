%% path

addpath(genpath('/home/vcc/3d_GAN_evaluation/matlab'))



% path_generatedSet
path_model = '/home/vcc/3d_GAN_evaluation/python/feature_ae/genereted_results/chair_with_4_parts/3D_gan_chair_32/';


% the saving path of all kinds of feature 
%notice that, path_cluster_adj_pair_t, path_cluster_adj_parts_t, path_cluster_component_t are the feature path of ground truth
%so ground truth features need to be saved in advance

path_adj_single = '../data/adj/single/generatedSet_32/chair_with_4_parts/3D_gan_chair_32/';

path_adj_pair = '../data/adj/pair/generatedSet_32/chair_with_4_parts/3D_gan_chair_32.mat';


path_adj_parts = '../data/adj/parts/generatedSet_32/chair_with_4_parts/3D_gan_chair_32.mat';
path_component = '../data/comp/generatedSet_32/chair_with_4_parts/3D_gan_chair_32.mat';


path_cluster_adj_pair = '../data/cluster/adj/pair/generatedSet_32/chair_with_4_parts/3D_gan_chair_32/3D_gan_chair_32.mat';
path_cluster_adj_pair_t = '../data/cluster/adj/pair/generatedSet_32/chair_with_4_parts/GT/GT.mat';

path_cluster_adj_parts = '../data/cluster/adj/parts/generatedSet_32/chair_with_4_parts/3D_gan_chair_32/3D_gan_chair_32.mat';
path_cluster_adj_parts_t = '../data/cluster/adj//generatedSet_32/chair_with_4_parts/GT/GT.mat';


path_cluster_component = '../data/cluster/comp/generatedSet_32/chair_with_4_parts/3D_gan_chair_32/3D_gan_chair_32.mat';
path_cluster_component_t = '../data/cluster/comp/generatedSet_32/chair_with_4_parts/GT/GT.mat';



% %% config
% dims -- the dim of 3D shape
% parts -- the amount of semantic parts
% type -- data type T-training data  G-generated data
% clusterNums -- the number of clusters
% gmm_count --the number of gaussian kernal
config = struct('dims',32,'parts',4,'type','G','clusterNums',[5,10,15,20,25,30,35,40],'gmm_count',10);



%extract part-wise, pair-wise and component-wisefeature
extract_adjacent(path_model,path_adj_single,path_adj_pair,path_adj_parts,config);
extract_component(path_model,path_component,config);

%cluster part-wise, pair-wise and component-wisefeature 
cluster_adj_pair(path_adj_pair,path_cluster_adj_pair,path_cluster_adj_pair_t,config); 
cluster_adj_parts(path_adj_parts,path_cluster_adj_parts,path_cluster_adj_parts_t,config);
cluster_component(path_component,path_cluster_component,path_cluster_component_t,config);




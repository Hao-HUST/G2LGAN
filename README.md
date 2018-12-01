## Global-to-Local Generative Model for 3D Shapes

### Introduction

Three-dimensional content creation has been a central research area in computer graphics for decades. The main challenge is to minimize manual intervention, while still allowing the creation of a variety of plausible 3D objects.In this work, we present a global-to-local generative model to synthesize 3D man-made shapes; see Fig. 1 as an example. It is based on an adversarial network to construct a global structure of the shape, with local part labels. The global discriminator is trained to distinguish between the whole real and generated 3D shapes, while the local discriminators focus on the individual local parts. A novel conditional auto-encoder is then introduced to enhance the part synthesis. 

![overview](overview.jpg)

### Usage

### License
Our code is released under MIT License (see LICENSE file for details).

### Citation

Please cite the paper in your publications if it helps your research:
```
@article{Hu18,
title = {Predictive and Generative Neural Networks for Object Functionality},
author = {Ruizhen Hu and Zihao Yan and Jingwen Zhan and Oliver van Kaick and Ariel Shamir and Hao Zhang and Hui Huang},
journal = {ACM Transactions on Graphics (Proc. SIGGRAPH)},
volume = {37},
number = {4},
pages = {151:1--151:14},  
year = {2018},
}
```

## Global-to-Local Generative Model for 3D Shapes

### Introduction

Three-dimensional content creation has been a central research area in computer graphics for decades. The main challenge is to minimize manual intervention, while still allowing the creation of a variety of plausible 3D objects.In this work, we present a global-to-local generative model to synthesize 3D man-made shapes.Our generative model consists of two units: a Global-to-Local GAN (G2LGAN) and a Part Refiner (PR).

G2LGAN consist of two networks: a **Global-to-Local GAN**(G2LGAN) and a **Part Refiner**(PR).

Specifically, using a generative adversarial network training, with global and local discriminators, out G2LGAN generates 3D semantically segmented models. The PR is based on an auto-encoder architecture and its goal is to refine the individual semantic parts output from G2LGAN.
For more details, please refer to our [paper](http://202.182.120.255/file/upload_file/image/research/att201810171620/G2L.pdf).

![overview](overview.jpg)

### Usage
The rest part of code will coming soon.
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

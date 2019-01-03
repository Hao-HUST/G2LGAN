## Global-to-Local Generative Model for 3D Shapes

### Introduction

Three-dimensional content creation has been a central research area in computer graphics for decades. The main challenge is to minimize manual intervention, while still allowing the creation of a variety of plausible 3D objects. In this work, we present a global-to-local generative model to synthesize 3D man-made shapes.

Global-to-Local Generative Model for 3D Shapes consist of two networks: a **Global-to-Local GAN**(G2LGAN) and a **Part Refiner**(PR).

Specifically, using a generative adversarial network training, with global and local discriminators, out **G2LGAN** generates 3D semantically segmented models. The **PR** is based on an auto-encoder architecture and its goal is to refine the individual semantic parts output from G2LGAN.
For more details, please refer to our [paper](http://202.182.120.255/file/upload_file/image/research/att201810171620/G2L.pdf).

![overview](overview.jpg)

### Usage
In our experiment, all the codes are tested under **Tensorflow 1.2.1** GPU version and **Python 2.7** on **Ubuntu 16.04**.

#### Train G2LGAN

To train a G2LGAN model to generate initial 3D shapes with semantic parts:

        cd G2LGAN
        python G2LGAN_train.py

To see all optional arguments for training:

        python G2LGAN_train.py -h
        
#### Generate Shapes from a pre-trained G2LGAN model

You can also sample some 3D shapes from pre-trained model:

        cd G2LGAN
        python G2LGAN_test.py 
        
#### Train Part Refiner

To train a Part Refiner model to refine the initial 3D shapes:

        cd Part_Refiner
        python train_part_refiner.py  
        
 #### Use the pre-trained Part Refiner to refine the initial shape

Based on the pre-trained PR model, you can refine the initial generated shapes from G2LGAN by:

        cd Part_Refiner
        python test_part_refiner.py         
        

#### Visulizaton Tool

If you want to visualize the trainig data(.mat files) or generated shapes(combined by a .npy files), you can use the Visulizaton Tool we provide, just run:

        cd visualization_tool
        python application.py

You can manually choose the shape file on the tool, and take the **visualization_tool/operation guide.pdf** as reference.   
        

### License
Our code is released under MIT License (see LICENSE file for details).

### Citation

Please cite the paper in your publications if it helps your research:
```
@article{G2L18,
title = {Global-to-Local Generative Model for 3D Shapes},
author = {Hao Wang and Nadav Schor and Ruizhen Hu and Haibin Huang and Daniel Cohen-Or and Hui Huang},
journal = {ACM Transactions on Graphics (Proc. SIGGRAPH ASIA)},
volume = {37},
number = {6},
pages = {214:1—214:10},  
year = {2018},
} 
```

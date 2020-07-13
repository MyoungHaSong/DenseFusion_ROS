# DenseFusion_ROS

This repository is based on https://github.com/j96w/DenseFusion and https://github.com/ooooverflow/BiSeNet.

If you are a person using Docker, https://hub.docker.com/repository/docker/choo2969/ros-densefusion

segmentation weight file [link](https://drive.google.com/drive/folders/1fRie5jwj9Liuwvs64_Mru8wUCy65Os0_?usp=sharing)
densefusion weight file [link](https://github.com/j96w/DenseFusion)

~~~
$ docker pull choo2969/ros-densefusion
~~~


## Requirements
---
- ROS (Kinetic)
- Python2.7
- Pytorch 0.4.1
- PIL
- scipy
- numpy
- pyyaml
- logging
- matplotlib
- CUDA



## Start
---
we have tested on Ubuntu 16.04 with ROS Kinetic and NVIDIA Titan XP and Geforce 1080 Ti 
1. Start camera node (D435)

    - Step1. Run your own camera, If your camera is not a D435 or D415, you will need to edit the RGB image and Depth Subscriber. Edit image_subscriber and depth_subscriber with your camera node
    ~~~
    vim path/densefusion/scripts/experiments/scripts/ros_eval_msg.sh
    ~~~
    
    - Step2. Edit the cam_cx,cam_cy,cam_fx,cam_fy values
    ~~~
    vim path/densefusion/scripts/tool/ros_eval_ycb_message.py
    ~~~

2. Start 
    ~~~
    sh path/densefusion/scripts/experiments/scripts/ros_eval_msg.sh
    ~~~
    Running this whill launch the SErvice Sever rining 6D Pose Estimation

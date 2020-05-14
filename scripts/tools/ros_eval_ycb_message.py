#! /usr/bin/env python

############# ros packages #####################
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from be.srv import AddTwoInts, AddTwoIntsResponse
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from geometry_msgs.msg import Pose, PoseArray
import tf
import message_filters

############ python pakcages ###################
import _init_paths
import argparse
import os
import copy
import random
import numpy as np
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from model.build_BiSeNet import BiSeNet
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
from matplotlib import pyplot as plt
import time



##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'datasets/ycb/YCB_Video_Dataset/', help='dataset root dir')
parser.add_argument('--model', type=str, default = 'trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = 'trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth',  help='resume PoseRefineNet model')
parser.add_argument('--checkpoint_path', type=str, default='trained_checkpoints/ycb/best_dice_loss.pth', help='The path to the pretrained weights of model')
parser.add_argument('--num_classes', type=int, default=21, help='num of object classes (with void)')
parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
parser.add_argument('--image_subscriber', type=str,defualt='/camera/color/image_raw')
parser.add_argument('--depth_subscriber', type=str,defualt='/camera/depth/image_rect_raw')



opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/ycb/Densefusion_iterative_result'
dist= np.array([0.0, 0.0, 0.0, 0.0, 0.0])


def image_callback(rgb):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(rgb,'bgr8')
    global cv_image


def depth_callback(depth):
    bridge = CvBridge()
    cv_depth = bridge.imgmsg_to_cv2(depth,'32SC1')
    global cv_depth

def rois_callback(rois):
    
    detect_res = rois.bounding_boxes
    global detect_res
    implimentation_seg()
    

rgb_sub = rospy.Subscriber(args.image_subsriber,Image, image_callback)
depth_sub = rospy.Subscriber(args.depth_subscriber,Image, depth_callback)
rois_sub = rospy.Subscriber('/darknet_ros/bounding_boxes',BoundingBoxes, rois_callback)
#########################################################################################

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

################################################################################################
"""
##################################################################################################
# get bbox coordinate
def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = 
    np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
"""
def get_bbox(rois,idx):
    # rmin = int(posecnn_rois[idx][2]) + 1
    # rmax = int(posecnn_rois[idx][4]) - 1
    # cmin = int(posecnn_rois[idx][1]) + 1
    # cmax = int(posecnn_rois[idx][3]) - 1
    rmin = int(rois[idx].xmin) + 1
    rmax = int(rois[idx].xmax) - 1
    cmin = int(rois[idx].ymin) + 1
    cmax = int(rois[idx].ymax) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
####################################################################################################
################################### load BiSeNet parameters ########################################
####################################################################################################
print('load BiseNet')
start_time = time.time()
bise_model = BiSeNet(opt.num_classes, opt.context_path)
bise_model = bise_model.cuda()
bise_model.load_state_dict(torch.load(opt.checkpoint_path))
global bise_model
print('Done!')
print("Load time : {}".format(time.time() - start_time))

#####################################################################################################
######################## load Densefusion Netwopy4thork, 3d model #############################
#####################################################################################################
print('load densefusion network')
start_time = time.time()
estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()
############################################################################
refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()
print('Done')
print("Load time : {}".format(time.time() - start_time))
#####################################################################################################
# class list upload
class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1
########################################################################################################
def seg_predict(image):
    global bise_model
    try:
        with torch.no_grad():
            bise_model.eval()
            h,w,_ = image.shape
            to_tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

            image = to_tensor(image)
            image = image.unsqueeze_(0)
            image = image.cuda()
            predict = bise_model(image).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict)
            predict = np.resize(predict,[h,w])
            print(np.unique(predict))
            zzzz = cv2.cvtColor(np.uint8(predict), cv2.COLOR_GRAY2BGR)
            cv2.imwrite('./segmentation_image.png', zzzz)

            return predict
    except CvBridgeError as e:
        print(e)





def pose_predict(img, depth,rois):
    class_list = ['002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '025_mug',
    '021_bleach_cleanser',
    '024_bowl',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker','051_large_clamp','052_extra_large_clamp','061_foam_brick']
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        object_number = len(rois)
        bridge = CvBridge()

        #lst = posecnn_rois[:,0:1].flatten()
        #lst = np.unique(label)
        my_result_wo_refine = []
        my_result = []
        for idx in range(object_number):
            #itemid = lst[idx]
            #itemid = class_list.index(rois[idx].Class) +1
            itemid = class_list.index(rois[idx].Class) +3
            
            try:
                label = seg_predict(img) 
                cv2.imwrite('/root/catkin_ws/src/dnsefusion/scripts/experiments/scripts/segmentation_image.png', label)
                rmin, rmax, cmin,cmax = get_bbox(rois,idx)
                # bounding box cutting
                #label = seg_predict(img[rmin:rmax,cmin:cmax,:]) 
                #mask_depth = ma.getmaskarray(ma.masked_not_equal(depth[rmin:rmax, cmin:cmax], 0))
                #mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
                #mask = mask_label * mask_depth
                # only image
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
                mask = mask_label * mask_depth
                #rmin, rmax, cmin, cmax = get_bbox(mask_label)

   
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                print(choose)
                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
                    
                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                pt2 = depth_masked / cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)
                img_masked = np.array(img)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()
                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
                pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
                pred_c = pred_c.view(bs, num_points)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(bs * num_points, 1, 3)
                points = cloud.view(bs * num_points, 1, 3)
                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)
                # making pose matrix
                dof = quaternion_matrix(my_r)
                dof[0:3,3] = my_t
                rot_to_angle = rotationMatrixToEulerAngles(dof[:3,:3])
                rot_to_angle = rot_to_angle.reshape(1,3)
                my_t = my_t.reshape(1,3)
                rot_t = np.concatenate([rot_to_angle,my_t], axis= 0)
                object_poses = {   
                    'tx':my_t[0][0],
                    'ty':my_t[0][1],
                    'tz':my_t[0][2],
                    'qx':my_r[0],
                    'qy':my_r[1],
                    'qz':my_r[2],
                    'qw':my_r[3]}
                my_result.append(object_poses)
                open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cam_mat = cv2.UMat(np.matrix([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy],
                             [0, 0, 1]]))
                imgpts, jac = cv2.projectPoints(cld[14], dof[0:3,0:3],dof[0:3,3],cam_mat,dist) # 14 mugcup
                open_cv_image = draw(open_cv_image,imgpts.get(), itemid)
                my_result_wo_refine.append(my_pred.tolist())
            except ZeroDivisionError:
                # my_result_wo_refine.append([0.0 for i in range(7)])
                # my_result.append([0.0 for i in range(7)])
                open_cv_image = None
                print('Fail')
    except CvBridgeError as e:
        print(e)
    
    return my_result, open_cv_image
    
def draw(img, imgpts, label):
    color = [[254,0,0],[254,244,0],[171,242,0],[0,216,254],[1,0,254],[95,0,254],[254,0,221],[0,0,0],[153,56,0],[138,36,124],[107,153,0],[5,0,153],[76,76,76],[32,153,67],[41,20,240],[230,111,240],[211,222,6],[40,233,70],[130,24,70],[244,200,210],[70,80,90],[30,40,30]]
    for point in imgpts:
        
        img=cv2.circle(img,(int(point[0][0]),int(point[0][1])), 1, color[int(label)], -1)
    return img 




    

def implimentation_seg():
    global cv_image
    global cv_depth
    global detect_res
    label_pub = rospy.Publisher('/label',Image, queue_size = 10)
    pose_pub = rospy.Publisher('/pose_pub', PoseArray,queue_size = 10)
    bridge = CvBridge()
    pose_fit_image = rospy.Publisher('/pose_fit_image_pub', Image, queue_size = 10)
    pose_estimation,fit_image = pose_predict(cv_image, cv_depth, detect_res)
    pose_array = PoseArray()
    pose_msg = Pose()
    print(pose_estimation)

    for i in range(len(pose_estimation)):
        pose_msg.position.x = pose_estimation[i]['tx']
        pose_msg.position.y = pose_estimation[i]['ty']
        pose_msg.position.z = pose_estimation[i]['tz']
        pose_msg.orientation.x = pose_estimation[i]['qx']
        pose_msg.orientation.y = pose_estimation[i]['qy']
        pose_msg.orientation.z = pose_estimation[i]['qz']
        pose_msg.orientation.w = pose_estimation[i]['qw']

        pose_array.poses.append(pose_msg)
    pose_pub.publish(pose_array)
    if fit_image is not None:
        pose_fit_image.publish(bridge.cv2_to_imgmsg(fit_image, 'bgr8'))


     
def main():

    rospy.init_node('pose_estimation_server')
    rospy.spin()

if __name__ == '__main__':
    main()

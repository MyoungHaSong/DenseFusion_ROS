#! /usr/bin/env python

##########################################  ros  packages ##############################################
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

########################################################################################################

import cv2
import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
from datasets.dataset import ycb_Dataset
from matplotlib import pyplot as plt
from torchvision import transforms


def eval(model,dataloader, args ):
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict)
            # predict = colour_code_segmentation(np.array(predict), label_info)

            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label)
            # label = colour_code_segmentation(np.array(label), label_info)

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
            save_img(i,data,predict)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]
        miou = np.mean(miou_list)
        print('IoU for each class:')
        tq.close()
        print('precision for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision
def save_img(iteration,img,label):
    img = img.cpu()
    img = img.numpy()
    img = np.transpose(img, [0,2,3,1])
    _,h,w,c = img.shape
    img = img.reshape([h,w,c])
    fig, axes = plt.subplots(1,2,figsize = (8,4))
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[1].imshow(label)
    plt.show()
    plt.savefig('./ycb/segmentation_result/{}.png'.format(iteration))
    plt.close()

######################################################################################################
############################################## test ##################################################
#####################################################################################################
class object_segmentation:
    def __init__(self,model):
        self.model = model
        self.bridge = CvBridge()
        self.label_pub = rospy.Publisher('label',Image,queue_size = 10)
        self.rgb_sub = rospy.Subscriber('rgb_image',Image, self.seg_callback)
    def seg_callback(self, rgb):
        try:
            with torch.no_grad():
                self.model.eval()
                rgb = self.bridge.imgmsg_to_cv2(rgb,'bgr8')
                self.to_tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                #rgb = np.transpose(rgb, (2,0,1))
                #rgb = np.expand_dims(rgb, axis = 0)
                #print(type(rgb))
                #rgb = torch.from_numpy(rgb)
                rgb = self.to_tensor(rgb)
                rgb = rgb.unsqueeze_(0)
                rgb = rgb.cuda()
                predict = self.model(rgb).squeeze()
                predict = reverse_one_hot(predict)
                predict = np.array(predict)
                np.save('./predict',predict)
                self.label_pub.publish(self.bridge.cv2_to_imgmsg(predict,'32SC1'))
                print('ss')
        except CvBridgeError as e:
            print(e)





def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')
    
    rospy.init_node('obj_seg',anonymous=True)
    Seg = object_segmentation(model)
    rospy.spin()
    


if __name__ == '__main__':
    params = [
        '--checkpoint_path', './checkpoints_18_sgd/best_dice_loss.pth',
        '--data', './CamVid/',
        '--cuda', '1',
        '--context_path', 'resnet101',
        '--num_classes', '21'
    ]
    main(params)

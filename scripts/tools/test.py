#! /usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class ImageIo:
    def __init__(self):
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw',Image, self.rgb_callback)
    def rgb_callback(self,rgb):
        bridge = CvBridge()
        label_pub = rospy.Publisher('/label',Image, queue_size = 10)
        img = bridge.imgmsg_to_cv2(rgb,"bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        label_pub.publish(bridge.cv2_to_imgmsg(image,encoding="8UC1"))





def main():
    IO = ImageIo()

if __name__ == '__main__':
    rospy.init_node('zzz',anonymous = True)
    main()
    rospy.spin()
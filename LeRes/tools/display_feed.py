import rospy
import sys
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import random 
from pipeline_test import pipe_inference, Pipe_model


pipe_model = Pipe_model()

def process_image(msg):
    #if count % 10 != 0:
    #    return
    #count
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    shape = cv_image.shape
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) 

    pred_depth = pipe_inference(pipe_model.model, cv_image)
    pred_depth_ori = cv2.resize(pred_depth, (shape[1], shape[0]))
    
    # time.sleep(0.1)

    # cv_image = (cv_image - np.min(cv_image)) / np.max(cv_image)
    # cv_image = (cv_image*255).astype(np.uint8)
    #print(np.max(pred_depth_ori), np.min(pred_depth_ori))
    pred_depth_ori = pred_depth_ori / np.max(pred_depth_ori)

    pdepth = (pred_depth_ori * 255).astype(np.uint8)
    cv_image2 = cv2.applyColorMap(pdepth, cv2.COLORMAP_JET)
    cv2.imshow("image",cv_image2)
    cv2.waitKey(50)

if __name__ == '__main__':
    while not rospy.is_shutdown():
        rospy.init_node('image_sub')
        rospy.loginfo('image_sub node started')
        rospy.Subscriber("/usb_cam/image_raw", Image, process_image)

        rospy.spin()

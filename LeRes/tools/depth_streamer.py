import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from pipeline_test import pipe_inference, Pipe_model

# SHIT FOR TEST: WILL FIX LATER
class DepthEstimator:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.count = 0
        
        self.pipe_model = Pipe_model()
        self.cv_bridge = CvBridge()
        self.spin_callback = rospy.Timer(rospy.Duration(0.1), self.spin)
        self.camera_image = np.zeros((self.height, self.width))
        self.pred_depth_uint8 = np.zeros((self.height, self.width)).astype(np.uint8)
        self.depth_color = np.zeros((self.height, self.width, 3))
        self.new_image = False
        self.new_bbox = False
        self.qr_depth = []
        self.image_streamer = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.image_callback
        )

        self.depth_publisher = rospy.Publisher ('/mono_depth/image_raw', Image, queue_size=10)


    def image_callback(self, msg):
        self.camera_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.new_image = True



    

    def spin(self, event):
        if self.new_image:
            self.new_image = False
            shape = self.camera_image.shape

            rgb_image = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2RGB)
            
            pred_depth = pipe_inference(self.pipe_model.model, rgb_image)

            pred_depth_resized = cv2.resize(pred_depth, (shape[1], shape[0]))

            scale = np.percentile(np.unique(np.around(pred_depth_resized, decimals=2)),90)

            pred_depth_scaled = pred_depth_resized / scale #np.max(pred_depth_resized)

            pred_depth_scaled = np.clip(pred_depth_scaled, 0, 1)

            self.pred_depth_uint8 = (pred_depth_scaled * 255).astype(np.uint8)

            # depth_image_color = cv2.applyColorMap(self.pred_depth_uint8, cv2.COLORMAP_JET)
            self.depth_publisher.publish(self.cv_bridge.cv2_to_imgmsg(self.pred_depth_uint8, encoding="passthrough"))



if __name__ == "__main__":
    rospy.init_node("depth_estimation", anonymous=True)
    my_depth_estimator = DepthEstimator()
    while not rospy.is_shutdown():
        rospy.spin()

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
import matplotlib.pyplot as plt
import numpy as np
import time
from pipeline_test import pipe_inference, Pipe_model


# SHIT FOR TEST: WILL FIX LATER
class DepthEstimator:
    def __init__(self):
        self.width = 1280
        self.height = 720
        
        self.pipe_model = Pipe_model()
        self.cv_bridge = CvBridge()
        self.spin_callback = rospy.Timer(rospy.Duration(0.1), self.spin)
        self.camera_image = np.zeros((self.height, self.width))
        self.pred_depth_uint8 = np.zeros((self.height, self.width)).astype(np.uint8)
        self.bboxes = BoundingBoxes()
        self.new_image = False
        self.new_bbox = False
        self.image_streamer = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.image_callback
        )
        self.bbox_streamer = rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_callback
        )

    def image_callback(self, msg):
        self.camera_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.new_image = True

    def darknet_callback(self, msg):
        self.bboxes = msg
        self.new_bbox = True

    def spin(self, event):
        if self.new_bbox:
            self.new_bbox = False
            all_bboxes = self.bboxes.bounding_boxes
            count = 0
            for box in all_bboxes:
                if box.Class != "person":
                    continue
                count += 1
                center_x = (box.xmin + box.xmax)//2
                center_y = (box.ymin + box.ymax)//2
                cur_depth = self.pred_depth_uint8[center_y, center_x]
                print(f"{box.Class}-{count}: {cur_depth}")
            print()



        if self.new_image:
            self.new_image = False
            shape = self.camera_image.shape

            rgb_image = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2RGB)

            pred_depth = pipe_inference(self.pipe_model.model, rgb_image)

            pred_depth_resized = cv2.resize(pred_depth, (shape[1], shape[0]))

            pred_depth_scaled = pred_depth_resized / np.max(pred_depth_resized)

            self.pred_depth_uint8 = (pred_depth_scaled * 255).astype(np.uint8)

            depth_image_color = cv2.applyColorMap(self.pred_depth_uint8, cv2.COLORMAP_JET)

            cv2.imshow("image", depth_image_color)
            cv2.waitKey(50)


if __name__ == "__main__":
    rospy.init_node("depth_estimation", anonymous=True)
    my_depth_estimator = DepthEstimator()
    while not rospy.is_shutdown():
        rospy.spin()

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2.aruco as aruco

# SHIT FOR TEST: WILL FIX LATER
class DepthEstimator:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.count = 0
        
        self.cv_bridge = CvBridge()
        self.spin_callback = rospy.Timer(rospy.Duration(0.1), self.spin)
        self.depth_color = np.zeros((self.height, self.width))
        self.camera_image = np.zeros((self.height, self.width))
        self.pred_depth_uint8 = np.zeros((self.height, self.width)).astype(np.uint8)
        self.depth_color = np.zeros((self.height, self.width, 3))
        self.bboxes = BoundingBoxes()
        self.new_camera_image = False
        self.new_depth_image = False
        self.new_bbox = False
        self.qr_depth = []
        self.depth_streamer = rospy.Subscriber(
            "/mono_depth/image_raw", Image, self.depth_callback
        )
        self.bbox_streamer = rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_callback
        )
        self.image_streamer = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.image_callback
        )

        self.valid_order = [1]
        self.time_length = 30

    def image_callback(self, msg):
        self.camera_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.new_camera_image = True
    def depth_callback(self, msg):
        self.pred_depth_uint8 = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.depth_color = cv2.cvtColor(self.pred_depth_uint8, cv2.COLOR_GRAY2RGB)
        self.new_depth_image = True
    def darknet_callback(self, msg):
        self.bboxes = msg
        self.new_bbox = True

    def get_qr_code_bboxes(self, rgb_image):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        aruco_params = aruco.DetectorParameters_create()
        (corners, ids, _) = aruco.detectMarkers(rgb_image, aruco_dict, parameters=aruco_params)
        self.qr_depth = []
        depth_0 = 1
        if len(corners)>0:
            ids = ids.flatten()
            arr = []
            for cur_corners, cur_id in zip(corners, ids):
                cur_corners = cur_corners.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = cur_corners
                # convert each of the (x, y)-coordinate pairs to integers
                # topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                # bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                center_row = (topLeft[1] + bottomRight[1]) // 2
                center_col = (topLeft[0] + bottomRight[0]) // 2

                qr_depth = np.nanmedian(self.pred_depth_uint8[center_row-4:center_row+4, center_col-4:center_col+4])

                if cur_id == 0:
                    depth_0 = qr_depth

                arr.append((cur_id, qr_depth, topLeft, bottomRight))
            arr.sort(key=lambda item: item[1])
            self.qr_depth = arr
            is_ordered = True
            last_id = -1
            for cur_id, qr_depth, topLeft, bottomRight in arr:
                is_ordered = is_ordered and (cur_id > last_id)
                last_id = cur_id
                # cv2.rectangle(self.depth_color, topLeft, bottomRight, (255, 0, 0), 1)
                qr_depth = round(qr_depth/depth_0, 2)
                cv2.putText(self.depth_color, f"{cur_id}: {qr_depth}", (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if len(self.valid_order) == self.time_length:
                self.valid_order.pop(0)
            self.valid_order.append(1 if is_ordered else 0)
            assert(len(self.valid_order) <= self.time_length)
            accuracy = sum(self.valid_order)/len(self.valid_order)
            print(f"Accuracy for last {self.time_length} frames: {accuracy}")
            cv2.putText(self.depth_color, f"Acc: {round(accuracy, 3)}", (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            

        if self.new_bbox:
            self.new_bbox = False
            for box in self.bboxes.bounding_boxes:
                subbox = False
                for box2 in self.bboxes.bounding_boxes:
                    if box2.xmin < box.xmin and box.xmax < box2.xmax and box2.ymin<box.ymin and box.ymax<box2.ymax: #check if box2 is superset of box
                        subbox=True
                        break
                if not subbox:
                    cv2.rectangle(self.depth_color, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 1)
                    cv2.putText(self.depth_color, f"{box.Class}", (box.xmin, box.ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imwrite("/shared/images/depth_image%04d.png" % self.count, self.depth_color)
        # cv2.imwrite("/shared/images/rgb_image%04d.png" % self.count, rgb_image)
        # self.count += 1 

    def spin(self, event):
        if self.new_depth_image:
            self.new_depth_image = False
            
            
            self.get_qr_code_bboxes(self.camera_image)
            # arr = []
            # for cur_id, qr_depth, _,_ in self.qr_depth:
            #     arr.append((cur_id, qr_depth))
            # print(arr)

            cv2.imshow("image", self.depth_color)
            cv2.waitKey(50)


if __name__ == "__main__":
    rospy.init_node("depth_estimation", anonymous=True)
    my_depth_estimator = DepthEstimator()
    while not rospy.is_shutdown():
        rospy.spin()

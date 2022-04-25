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
import cv2.aruco as aruco

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
        self.depth_color = np.zeros((self.height, self.width, 3))
        self.bboxes = BoundingBoxes()
        self.new_image = False
        self.new_bbox = False
        self.qr_depth = []
        self.image_streamer = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.image_callback
        )
        self.bbox_streamer = rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_callback
        )
        self.depth_publisher = rospy.Publisher ('mono_depth', Image, queue_size=10)

    def image_callback(self, msg):
        self.camera_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.new_image = True

    def darknet_callback(self, msg):
        self.bboxes = msg
        self.new_bbox = True

    def get_qr_code_bboxes(self, rgb_image):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        aruco_params = aruco.DetectorParameters_create()
        (corners, ids, _) = aruco.detectMarkers(rgb_image, aruco_dict, parameters=aruco_params)
        self.qr_depth = []
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

                arr.append((cur_id, qr_depth, topLeft, bottomRight))
            arr.sort(key=lambda item: item[1])
            self.qr_depth = arr
            for cur_id, qr_depth, topLeft, bottomRight in arr:
                cv2.rectangle(self.depth_color, topLeft, bottomRight, (255, 0, 0), 2)
                cv2.putText(self.depth_color, f"{cur_id}: {qr_depth}", (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for box in self.bboxes.bounding_boxes:
            cv2.rectangle(self.depth_color, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 2)
            cv2.putText(self.depth_color, f"{box.Class}", (box.xmin, box.ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def spin(self, event):
        if False and self.new_bbox:
            self.new_bbox = False
            all_bboxes = self.bboxes.bounding_boxes
            count = 0
            objs = list(filter(lambda box: box.Class == "person", all_bboxes))
            objs.sort(key=lambda box: box.xmax - box.xmin)

            if not objs:
                return

            box = objs[0]

            center_x = (box.xmin + box.xmax)//2
            center_y = (box.ymin + box.ymax)//2
            cur_depth = np.median(self.pred_depth_uint8[center_y-10:center_y+10, center_x-10:center_x+10])
            print(f"{box.Class}-{count}: {cur_depth}")

            print(f"qr_depth = {self.qr_depth}")
            print("depth: ", cur_depth/self.qr_depth)
            # for box in all_bboxes:
            #     if box.Class != "bottle":
            #         continue

                

            #     if box.Class != "person":
            #         continue
            #     count += 1
            #     center_x = (box.xmin + box.xmax)//2
            #     center_y = (box.ymin + box.ymax)//2
            #     cur_depth = self.pred_depth_uint8[center_y, center_x]
            #     print(f"{box.Class}-{count}: {cur_depth}")
            print()



        if self.new_image:
            self.new_image = False
            shape = self.camera_image.shape

            rgb_image = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2RGB)
            
            pred_depth = pipe_inference(self.pipe_model.model, rgb_image)

            pred_depth_resized = cv2.resize(pred_depth, (shape[1], shape[0]))

            pred_depth_scaled = pred_depth_resized / np.max(pred_depth_resized)

            pred_depth_scaled = np.clip(pred_depth_scaled, 0, 1)

            self.pred_depth_uint8 = (pred_depth_scaled * 255).astype(np.uint8)
            self.depth_color = cv2.cvtColor(self.pred_depth_uint8, cv2.COLOR_GRAY2RGB)

            # depth_image_color = cv2.applyColorMap(self.pred_depth_uint8, cv2.COLORMAP_JET)
            # self.depth_publisher.publish(self.cv_bridge.cv2_to_imgmsg(self.pred_depth_uint8, encoding="passthrough"))

            
            self.get_qr_code_bboxes(rgb_image)
            arr = []
            for cur_id, qr_depth, _,_ in self.qr_depth:
                arr.append((cur_id, qr_depth))
            print(arr)

            cv2.imshow("image", self.depth_color)
            cv2.waitKey(50)


if __name__ == "__main__":
    rospy.init_node("depth_estimation", anonymous=True)
    my_depth_estimator = DepthEstimator()
    while not rospy.is_shutdown():
        rospy.spin()

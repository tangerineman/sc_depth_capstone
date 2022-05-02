import rospy
import numpy as np 
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from darknet_ros_msgs.msg import BoundingBoxes
import matplotlib.pyplot as plt
import numpy as np
import time
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


# SHIT FOR TEST: WILL FIX LATER
class DepthEstimator: 
    def __init__(self):
        self.depth_image = cv2.imread("/shared/sc_depth_capstone/LeRes/test_images/5.jpg")

        self.imageWidth = 1280
        self.imageHeight = 720

        self.pipe_model = Pipe_model()

        # self.depth_image = np.zeros((self.imageWidth, self.imageHeight))
        self.depth_camera_info = CameraInfo()

        # self.depth_image_streamer = rospy.Subscriber("/usb_cam/image_raw", Image, self.depth_callback)
        # self.depth_camera_info = rospy.Subscriber("/usb_cam/camera_info", CameraInfo, self.camera_info_callback)

        self.cv_bridge = CvBridge()

        self.spin_callback = rospy.Timer(rospy.Duration(5), self.spin)
        self.new_image = True
        depth_model = RelDepthModel(backbone=args.backbone)
        depth_model.eval()

        # load checkpoint
        load_ckpt(args, depth_model, None, None)
        depth_model.cuda()


    def camera_info_callback(self, msg):
      self.depth_camera_info = msg

    def depth_callback(self, msg):
      # self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
      # self.depth_image = cv2.imread("/shared/sc_depth_capstone/LeRes/test_images/5.jpg")
      self.new_image=True


    def spin(self, event):
      if(self.new_image):
        rgb = self.depth_image
        rgb_c = rgb[:, :, ::-1].copy()
        gt_depth = None
        A_resize = cv2.resize(rgb_c, (448, 448))
        rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        img_torch = scale_torch(A_resize)[None, :, :, :]
        start = time.time()
        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        end = time.time()
        print("TIME ELAPSED:", end-start)

        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        # if GT depth is available, uncomment the following part to recover the metric depth
        #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

        img_name = v.split('/')[-1]
        cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        # save depth
        plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
        cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
 

        cv2.imshow("image", cv_image)
        cv2.waitKey(50)
          


       
      


if __name__ == "__main__":

    rospy.init_node('depth_estimation', anonymous=True)
    my_depth_estimator = DepthEstimator()
    while not rospy.is_shutdown():
        rospy.spin()

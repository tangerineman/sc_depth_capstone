from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import argparse
import numpy as np
import torch
import time

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

class MyArgs:
    backbone = "resnet50"
    load_ckpt = "res50.pth"

def pipe_inference(depth_model, rgb):
    rgb_c = rgb[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (448, 448))

    img_torch = scale_torch(A_resize)[None, :, :, :]

    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    return pred_depth

class Pipe_model:
    def __init__(self):
        args = MyArgs()
        depth_model = RelDepthModel(backbone=args.backbone)
        depth_model.eval()

        # load checkpoint
        load_ckpt(args, depth_model, None, None)
        depth_model.cuda()

        self.model = depth_model

"""
if __name__ == '__main__':

    args = parse_args()

    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()

    image_dir = os.path.dirname(os.path.dirname(__file__)) + '/test_images/'
    imgs_list = os.listdir(image_dir)
    imgs_list.sort()
    imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
    image_dir_out = image_dir + '/outputs'
    os.makedirs(image_dir_out, exist_ok=True)

    for i, v in enumerate(imgs_path):
        print('processing (%04d)-th image... %s' % (i, v))
        rgb = cv2.imread(v)
        

        
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        # if GT depth is available, uncomment the following part to recover the metric depth
        #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

        #img_name = v.split('/')[-1]
        #cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        # save depth
        #plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
        #cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
    #torch.onnx.export(depth_model.depth_model.cpu(), img_torch, "LeRes.onnx", verbose=False, opset_version=11)
    """
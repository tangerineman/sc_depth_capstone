import onnxruntime as nxrun
import numpy as np

from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
import os
import cv2

from config import get_opts

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2

import datasets.custom_transforms as custom_transforms
import torchvision.transforms as transforms

from visualization import *

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

# normaliazation
inference_transform = custom_transforms.Compose([
    custom_transforms.RescaleTo([448, 448]),
    custom_transforms.ArrayToTensor(),
    custom_transforms.Normalize()]
)

sess = nxrun.InferenceSession("LeRes.onnx")

# data = np.load('test.npy')
# result = sess.run(None, {input_name: data})
# print(result[0].argmax())
# print(result[0])
# from PIL import Image
#Image.fromarray(np.load('test.npy').reshape(28, 28) * 255).show()

input_name = sess.get_inputs()[0].name

rgb = cv2.imread("demo/input/LeRes-test.jpg")
rgb_c = rgb[:, :, ::-1].copy()
gt_depth = None
A_resize = cv2.resize(rgb_c, (448, 448))
rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

img = scale_torch(A_resize)[None, :, :, :]
print(img.shape)
#img = imread("demo/input/5.jpg").astype(np.float32)
#tensor_img = inference_transform([img[0]])[0][0].unsqueeze(0).cuda()
tensor_img = img.cpu().detach().numpy()
np.save("LeRes_test", tensor_img)

result = sess.run(None, {input_name: tensor_img})
result = torch.FloatTensor(result)[0]
#result = torch.clamp(result, min=0.0, max=1.0)

print(result[0].argmax())
print(result[0])

vis = visualize_depth(result[0, 0]).permute(1, 2, 0).numpy() * 255
        
imwrite('onnx_test_depth-2.jpg', vis.astype(np.uint8))

depth = result[0, 0].cpu().numpy()
np.save('onnx_test_depth.npy', depth)

from PIL import Image
#Image.fromarray(np.load('test.npy').reshape(28, 28) * 255).show()
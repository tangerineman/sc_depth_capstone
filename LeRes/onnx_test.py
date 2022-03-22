import onnxruntime as nxrun
import numpy as np

from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
import os

from config import get_opts

import datasets.custom_transforms as custom_transforms

from visualization import *

# normaliazation
inference_transform = custom_transforms.Compose([
    custom_transforms.RescaleTo([256, 320]),
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

img = imread("test_images/5.jpg").astype(np.float32)
tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
tensor_img = tensor_img.cpu().detach().numpy()

result = sess.run(None, {input_name: tensor_img})
result = torch.FloatTensor(result)[0]
result = torch.clamp(result, min=0.0, max=1.0)
print("TYPPPPE", type(result), result.size(), result.dtype)

print(result[0].argmax())
print(result[0])

vis = visualize_depth(result[0, 0]).permute(1, 2, 0).numpy() * 255
        
imwrite('onnx_test_depth.jpg', vis.astype(np.uint8))

depth = result[0, 0].cpu().numpy()
np.save('onnx_test_depth.npy', depth)

from PIL import Image
#Image.fromarray(np.load('test.npy').reshape(28, 28) * 255).show()
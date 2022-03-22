import numpy as np
import matplotlib.pyplot as plt

onnx_v = 'onnx_test_depth.npy'
base = 'demo/output/model_v2/depth/00.npy'

plt.figure()
plt.hist(np.load(onnx_v).flatten(), bins=20)
plt.figure()
plt.hist(np.load(base).flatten(), bins=20)

plt.show()
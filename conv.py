import glob
import numpy as np
from PIL import Image
filelist = glob.glob('/home/rk010/Extended_SURE-pytorch_version/data_old/RES180/*.jpg')

x = np.array([np.array(Image.open(fname)) for fname in filelist])

x = x[ :, :, :,1]

x = np.array(x).astype(np.float32) + np.array([np.random.normal(0, 0.04*255.0, (180,180))], dtype='float32')

#x = (np.array(x)-np.min(x))/(np.max(x)-np.min(x))

print(x.shape, np.max(x))

x.dump("/home/rk010/Restart/Extended_SURE-pytorch_version/data_old/CT_noisy0.05_180x180.npy")
x.dump("/home/rk010/Extended_SURE-pytorch_version/data_old/CT_noisy0.05_180x180.npy")
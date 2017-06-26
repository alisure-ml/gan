import numpy as np
import tensorflow as tf
import os


# 产生[-1,1)的均匀分布的随机数组[m,n]
def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# 为了展示
def sample_y(m, n, ind=4):
    y = np.zeros([m, n])
    for i in range(m):
        y[i, i % ind] = 1
    return y


# 将z和y沿着第1维拼接
def concat(z, y):
    return tf.concat([z, y], 1)


# 判定是否输出图像
def save_imgae(epoch):
    result = epoch % 1024 == 0 or (15000 < epoch < 30000 and epoch % 512 == 0) \
             or (2048 < epoch < 15000 and epoch % 256 == 0) or (2048 < epoch < 4096 and epoch % 128 == 0) \
             or (1024 < epoch < 2048 and epoch % 64 == 0) or (512 < epoch < 1024 and epoch % 32 == 0) \
             or (256 < epoch < 512 and epoch % 16 == 0) or (128 < epoch < 256 and epoch % 10 == 0) \
             or (64 < epoch < 128 and epoch % 6 == 0) or (32 < epoch < 64 and epoch % 4 == 0) \
             or (8 < epoch < 32 and epoch % 2 == 0) or (epoch <= 8)
    return result


# 新建目录
def make_dir_if_noe_exist(sample_dir):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    pass

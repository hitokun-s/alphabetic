#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lib import *

using_gpu = False
xp = np
try:
    cuda.check_cuda_available()
    xp = cuda.cupy
    cuda.get_device(0).use()
    using_gpu = True
except:
    print  "I'm sorry. Using CPU."

image_dir = './images48'
out_image_dir = './alphabet_output'
out_model_dir = './alphabet_model'

nz = 100  # # of dim for Z
z_sample_size = 100
t_sample_size = 52
n_epoch = 200
output_interval = 20

fs = os.listdir(image_dir)
print len(fs)
dataset = []
for fn in fs:
    f = open('%s/%s' % (image_dir, fn), 'rb')
    img_bin = f.read()
    dataset.append(img_bin)
    f.close()
print len(dataset)

# sizeは、1次元なら数字、多次元ならタプル
def generate_rand(low, hight, size, dtype=np.float32):
    global using_gpu
    if using_gpu:
        # この書き方だと、xp = cuda.cupy なら良いが、xp = np の場合にエラーになる
        generated = xp.random.uniform(low, hight, size, dtype=dtype)
    else:
        generated = xp.random.uniform(low, hight, size).astype(dtype)
    return generated

def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))

def binarize(ndArr, th=50):
    (rowCnt, colCnt) = ndArr.shape
    for i in range(rowCnt):
        for j in range(colCnt):
            ndArr[i][j] = -1 if ndArr[i][j] > th else 1 # 白（背景）を-1に、黒を1にしている
    return ndArr

zvis = (generate_rand(-1, 1, (100, nz), dtype=np.float32))

# サンプル52にしたことだし、もう固定で作ってしまう
# ----------------------------------------------------------------------------------
x2 = np.zeros((52, 1, 48, 48), dtype=np.float32)
for j in range(52):
    res = np.asarray(Image.open(StringIO(dataset[j])).convert('L')).astype(np.float32)
    res.flags.writeable = True
    x2[j, :, :, :] = binarize(res, 50)

print x2[0][0]

import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import label, regionprops

tgt = x2[35][0]

all_labels = measure.label(tgt)
print all_labels.shape
blobs_labels = measure.label(tgt, background=0)
print blobs_labels.shape

plt.figure(figsize=(12, 3.5))
plt.subplot(151)
plt.imshow(tgt, cmap='gray')
plt.axis('off')
plt.subplot(152)
plt.imshow(all_labels, cmap='spectral')
plt.axis('off')
plt.subplot(153)
plt.imshow(blobs_labels, cmap='spectral')
plt.axis('off')

# regionがもつプロパティの詳細
# http://www.mathworks.com/help/images/ref/regionprops.html

regions = regionprops(label(tgt))
for region in regions:
    print "=========================="
    print region.area # 面積（含まれる画素数）
    print region.centroid # 中心座標
    print region.perimeter # 周長
    print region.euler_number
    print "circularity:%s" % (region.area / region.perimeter**2)
    print "complexity:%s" % (region.perimeter**2 / region.area)

plt.tight_layout()
# plt.show()
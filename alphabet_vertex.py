#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from lib import *
from util import *

image_dir = './images48'
out_model_dir = checked("./alphabet_model")
vertex_image__dir = checked("./alphabet_vertex")

logger = getFileLogger("alphabet_vertex")
logger.info("Start!")

using_gpu = False
xp = np
try:
    cuda.check_cuda_available()
    xp = cuda.cupy
    cuda.get_device(0).use()
    using_gpu = True
except:
    logger.info("I'm sorry. Using CPU.")

nz = 100

# load sample ================================================================
fs = sorted(os.listdir(image_dir)) # アルファベット順にしておく
assert len(fs) == 52
dataset = []
for fidx, fn in enumerate(fs):
    f = open('%s/%s' % (image_dir, fn), 'rb')
    logger.info("file name:%s, idx:%d" % (fn, fidx))
    img_bin = f.read()
    dataset.append(img_bin)
    f.close()
sample_dict = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def binarize(ndArr, th=50):
    (rowCnt, colCnt) = ndArr.shape
    for i in range(rowCnt):
        for j in range(colCnt):
            ndArr[i][j] = -1 if ndArr[i][j] > th else 1  # 白（背景）を-1に、黒を1にしている
    return ndArr


x2 = np.zeros((52, 1, 48, 48), dtype=np.float32)
for j in range(52):
    res = np.asarray(Image.open(StringIO(dataset[j])).convert('L')).astype(np.float32)
    res.flags.writeable = True
    x2[j, :, :, :] = binarize(res, 50)
# =============================================================================

# choose 1 sample
# sample = x2[0]
# sample = Variable(sample, volatile=True)
# sample = Variable(cuda.to_gpu(sample) if using_gpu else sample)
# ================================================================================

# load model

gen = Generator(nz=nz)
o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)

o_gen.setup(gen)
o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))

if using_gpu:
    gen.to_gpu()

serializers.load_hdf5("%s/dcgan_model_gen.h5" % out_model_dir, gen)
serializers.load_hdf5("%s/dcgan_state_gen.h5" % out_model_dir, o_gen)

# Funcs =============================================================================

def save_image(img, new_w, new_h, it):
    im = np.zeros((new_h, new_w, 3))
    im[:, :, 0] = x[2, :, :]
    im[:, :, 1] = x[1, :, :]
    im[:, :, 2] = x[0, :, :]

    def clip(a):
        return 0 if a < 0 else (255 if a > 255 else a)

    im = np.vectorize(clip)(im).astype(np.uint8)
    Image.fromarray(im).save(vertex_image__dir + "/im_%05d.png" % it)

class Clip(chainer.Function):
    def forward(self, x):
        x = x[0]
        ret = cuda.elementwise(
                'T x','T ret',
                '''
                    ret = x<-1?-1:(x>1?1:x);
                ''','clip')(x)
        return ret
def clip(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))

# sizeは、1次元なら数字、多次元ならタプル
def generate_rand(low, hight, size, dtype=np.float32):
    global using_gpu
    if using_gpu:
        # この書き方だと、xp = cuda.cupy なら良いが、xp = np の場合にエラーになる
        generated = xp.random.uniform(low, hight, size, dtype=dtype)
    else:
        generated = xp.random.uniform(low, hight, size).astype(dtype)
    return generated

# test =========================================================
logger.info("Let's do test!")

def get_winner(z_ori, sample):
    # z[50:, :] = (xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
    z = Variable(z_ori)
    x = gen(z, test=True)
    x = x.data.get()

    # find the nearest gen image against teh sample
    nearest_idx = np.argmin([((sample - tgt) * (sample - tgt)).sum() for tgt in x])
    min_error = ((sample - x[nearest_idx]) * (sample - x[nearest_idx])).sum()

    best_z = z_ori[nearest_idx] # 生成パラメータ
    best_img = x[nearest_idx] # その結果画像

    return best_z, best_img, min_error

y = []
for sample_idx, sample in enumerate(x2):
    logger.info("let's go to sample:%d" % sample_idx)
    min_error = sys.maxint
    for it in range(100):
        logger.info("iterator:%d" % it)
        z = (generate_rand(-1, 1, (5000, nz), dtype=np.float32)) # ランダムな生成画像5000個から近いものを探す
        tmp_best_z, tmp_best_img, tmp_min_error = get_winner(z, sample)
        if tmp_min_error < min_error:
            logger.info("win!")
            best_z = tmp_best_z
            best_img = tmp_best_img
            min_error = tmp_min_error

    # 頂点候補に対して、生成パラメータを振動させ、さらに最近値を取得する
    # 振動幅をだんだん細かく
    for band in (np.arange(0.5, 0.1, -0.001).tolist() + np.arange(0.1, 0, -0.0002).tolist()):
        logger.info("post-battle by band:%s" % band)
        z_neighbors = (generate_rand(-band, band, (5000, nz), dtype=np.float32)) + best_z
        best_z2, best_img2, min_error2 = get_winner(z_neighbors, sample)
        if min_error2 < min_error:
            logger.info("new king showed up! min error:%s" % min_error2)
            best_z = best_z2
            best_img = best_img2
            min_error = min_error2

    y.append(best_img)
    logger.info(xp.array(best_z))
    logger.info("min error for sample:%s is:%s" % (sample_dict[sample_idx], min_error))
    xp.save("%s/%s.npy" % (vertex_image__dir, sample_dict[sample_idx]), cuda.to_cpu(xp.array(best_z)))

pylab.rcParams['figure.figsize'] = (16.0, 16.0)
pylab.clf()
y = xp.array(y)
logger.info(y.shape)
for i_ in range(52):
    _tmp = cuda.to_cpu(y[i_, 0, :, :])
    # if using_gpu:
    #     tmp = Clip().forward(y[i_, 0, :, :]).get()
    # else:
    #     tmp = np.vectorize(clip)(y[i_, 0, :, :])
    tmp = np.vectorize(clip)(_tmp)
    pylab.subplot(10, 10, i_ + 1)
    pylab.gray()
    pylab.imshow(tmp)
    pylab.axis('off')
pylab.savefig('%s/test.png' % (vertex_image__dir))
#===============================================================


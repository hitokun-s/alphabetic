#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random as rd
import sys

from skimage.measure import regionprops, label

from lib import *
from util import *

out_model_dir = checked("./alphabet_model")
vertex_image_dir = checked("./alphabet_vertex")
gen_image_dir = checked("./alphabet_gen")

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
    print  "I'm sorry. Using CPU."

nz = 100

# load model =======================================================================

gen = Generator(nz=nz)
o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
o_gen.setup(gen)
o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))

if using_gpu:
    gen.to_gpu()

serializers.load_hdf5("%s/dcgan_model_gen.h5" % out_model_dir, gen)
serializers.load_hdf5("%s/dcgan_state_gen.h5" % out_model_dir, o_gen)

# load gen params ================================================================

gen_params = {}

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# letters = "abcdefghijklmnopqrstuvwxyz"
for letter in letters:
    z = xp.load("%s/%s.npy" % (vertex_image_dir, letter))
    gen_params[letter] = z


# Funcs =============================================================================

class Clip(chainer.Function):
    def forward(self, x):
        x = x[0]
        ret = cuda.elementwise(
                'T x', 'T ret',
                '''
                    ret = x<-1?-1:(x>1?1:x);
                ''', 'clip')(x)
        return ret


def clip(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))


# gen =========================================================
print "Let's generate!"

# target_titles = [
#     "A", "B", "A+B", "A+B+C"
# ]
target_titles = [
    "S+A",
    "W+O+Y",
    "Z+E+N",
    "V+O+W",
    "S+G+A",
    "I+G+E",
    "G+E+N",
    "J+B+S",
    "K+Z+U",
    "S+J+K",
    "v+s+f",
    "W+U+Z+C",
    "B+U+X+A",
    "V+O+E+F",
    "U+E+V+T",
    "W+Q+A+X",
    "K+T+V+W",
    "H+P+U+F",
    "H+P+I+V",
    "X+H+M+A",
    "K+D+F+R"
]



def conv_to_params(title):
    if title.count("+"):
        titles = title.split("+")
        if len(titles) == 4:
            return gen_params[titles[0]] * 0.4 + gen_params[titles[1]] * 0.3 + gen_params[titles[2]] * 0.2 + gen_params[titles[3]] * 0.1
        else:
            # gen_params[t]は、cupyのndarrayになっているっぽい
            # np.mean, cupy.meanの方法がうまくいかないので、愚直に平均を出す
            res = xp.zeros(100, dtype=np.float32)
            for t in titles:
                res = res + gen_params[t]
            return res / len(titles)
    else:
        return gen_params[title]


targets = map(conv_to_params, target_titles)

gen_param = xp.array(targets, dtype=np.float32)
z = Variable(gen_param)
x = gen(z, test=True)
x = x.data.get()  # 生成画像リスト

pylab.rcParams['figure.figsize'] = (16.0, 16.0)
pylab.clf()
y = xp.array(x)
print y.shape


def binarize(ndArr, th=0):
    (rowCnt, colCnt) = ndArr.shape
    for i in range(rowCnt):
        for j in range(colCnt):
            ndArr[i][j] = -1 if ndArr[i][j] > th else 1 # 白（背景）を-1に、黒を1にしている
    return ndArr


# 白黒反転
def invert(binaryArr):
    f = np.vectorize(lambda x : 1 - x) # 全要素に作用する関数を作成（xは各要素の値）
    return f(binaryArr)


# 白黒反転すべきか
def should_invert(ndArr):
    assert ndArr.shape == (48,48)
    arr = []
    for i,v in np.ndenumerate(ndArr):
        if i[0] in [0,47] or i[1] in [0,47]:
            arr.append(v)
    arr = np.array(arr)
    # 外周要素の半数以上が１なら反転の必要がある
    return np.count_nonzero(arr) > arr.size / 2


def analyze(tgt, idx):
    print "%d :==========================" % idx
    tgt = binarize(tgt)
    if should_invert(tgt):
        tgt = invert(tgt)
    regions = regionprops(label(tgt))
    for region in regions:
        print "-----"
        print "area:%s" % region.area # 面積（含まれる画素数）
        print "centroid:" + str(region.centroid) # 中心座標
        print "perimeter:%s" % region.perimeter # 周長
        print "euler:%s" % region.euler_number
        print "circularity:%s" % (region.area / region.perimeter**2)
        print "complexity:%s" % (region.perimeter**2 / region.area)


for i_ in range(len(targets)):
    _tmp = cuda.to_cpu(y[i_, 0, :, :])
    # if using_gpu:
    #     tmp = Clip().forward(y[i_, 0, :, :]).get()
    # else:
    #     tmp = np.vectorize(clip)(y[i_, 0, :, :])
    tmp = np.vectorize(clip)(_tmp)
    analyze(tmp, i_)
    pylab.subplot(5, 5, i_ + 1)
    pylab.title(target_titles[i_], fontsize=20, fontweight='bold')
    # バグっぽい動き
    # if (i_ + 1) % 3 == 0:
    #     pylab.imshow(tmp, cmap="hot")
    # else:
    #     pylab.imshow(tmp, cmap="gray")
    color = "hot" if (i_ + 1) % 4 == 0 else "gray"
    pylab.imshow(tmp, cmap=color)
    pylab.axis('off')
pylab.savefig('%s/sample.png' % (gen_image_dir))
# ===============================================================

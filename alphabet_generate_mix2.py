#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from random import uniform, random, choice, sample
import random as rd
import sys

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

# letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
target_titles = []

for i in range(27):
    # ランダムに２つのアルファベット（例："S","M"）を選んで
    choices = rd.sample(letters, 2)
    # "S","M","S+M",という３つの文字列を追加する
    target_titles += [choices[0], choices[1], "+".join(choices)]


def conv_to_params(title):
    if title.count("+"):
        titles = title.split("+")
        # gen_params[t]は、cupyのndarrayになっているっぽい
        # np.mean, cupy.meanの方法がうまくいかないので、愚直に平均を出す
        res = xp.zeros(100, dtype=np.float32)
        for t in titles:
            res = res + gen_params[t]
        print res / len(titles)
        return res / len(titles)
        # return (gen_params[titles[0]] + gen_params[titles[1]]) / 2
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
for i_ in range(len(targets)):
    _tmp = cuda.to_cpu(y[i_, 0, :, :])
    # if using_gpu:
    #     tmp = Clip().forward(y[i_, 0, :, :]).get()
    # else:
    #     tmp = np.vectorize(clip)(y[i_, 0, :, :])
    tmp = np.vectorize(clip)(_tmp)
    pylab.subplot(9, 9, i_ + 1)
    pylab.title(target_titles[i_])
    # バグっぽい動き
    # if (i_ + 1) % 3 == 0:
    #     pylab.imshow(tmp, cmap="hot")
    # else:
    #     pylab.imshow(tmp, cmap="gray")
    color = "hot" if (i_ + 1) % 3 == 0 else "gray"
    pylab.imshow(tmp, cmap=color)
    pylab.axis('off')
pylab.savefig('%s/mix2.png' % (gen_image_dir))
# ===============================================================

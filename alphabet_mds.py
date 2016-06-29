#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 参考：http://blog.cocomoff.info/archives/2014/12/11/7121/
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, decomposition, manifold

from lib import *
from util import checked, getFileLogger

vertex_image_dir = checked("./alphabet_vertex") # 生成パラメータファイル（.npy）置き場

logger = getFileLogger("alphabet_pca")
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

# load gen params ================================================================
gen_params = {}
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for letter in letters:
    z = xp.load("%s/%s.npy" % (vertex_image_dir, letter))
    gen_params[letter] = cuda.to_cpu(z)

lower_letters = "abcdefghijklmnopqrstuvwxyz"
for lower_letter in lower_letters:
    z = xp.load("%s/lowercase/%s.npy" % (vertex_image_dir, lower_letter))
    gen_params[lower_letter] = cuda.to_cpu(z)

letters = letters + lower_letters
print letters
gen_params_arr = []
for l in letters:
    gen_params_arr.append(gen_params[l])

gen_params = np.array(gen_params_arr)
# 平均との差分をとっても結果は同じ
# avg = np.average(gen_params, 0)
# gen_params = gen_params - avg
print gen_params

# データセットの読み込み
X = gen_params # １サンプル＝（-1..+1）の範囲の100個の数値、それが52サンプルある
Y = letters
print Y # クラスインデックス（＝ラベル数字）の配列

# 主成分分析前のサイズ
print X.shape

# 多次元尺度法（MDS）による次元削減 =================================

# def dist(a,b):
#     return np.sum((a - b) ** 2)

data = np.array([np.sum((gen_params - x)**2, axis=1) for x in gen_params])

mds = manifold.MDS(n_components = 2, dissimilarity='precomputed')
X_mds = mds.fit_transform(data) # dataは距離行列

# ===================================================================

# 分析後のサイズ
print X_mds.shape # (52, 2)

# 可視化
# s  = np.array([x for i, x in enumerate(X_pca) if Y[i] == 0])
# ve = np.array([x for i, x in enumerate(X_pca) if Y[i] == 1])
# vi = np.array([x for i, x in enumerate(X_pca) if Y[i] == 2])
# colors = ['b.', 'r.', 'k.']
# fig, ax = plt.subplots()
# ax.plot(s[:,0],  s[:,1],  'b.', label='Setosa')
# ax.plot(ve[:,0], ve[:,1], 'r.', label='Versicolour')
# ax.plot(vi[:,0], vi[:,1], 'k.', label='Virginica')

# title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
#               'verticalalignment':'bottom'}

fig, ax = plt.subplots()
for i, d in enumerate(X_mds):
    ax.plot(d[0],  d[1],  'b.')
    ax.text(d[0],  d[1], letters[i], ha = 'center', va = 'bottom', size = 20, color="red")

ax.set_title("MDS for alphabet")

ax.legend(numpoints=1)

plt.savefig("alphabet_mds.png")

# test_lk_pyramid.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

from LucasKanadePyramid import LucasKanadePyramid
from file_utils import mkdir_if_missing

# Usage: python test_lk_pyramid.py car1 0
data_name  = sys.argv[1] if len(sys.argv)>1 else 'car1'
do_display = (not int(sys.argv[2])) if len(sys.argv)>2 else 1

# Load video frames
data = np.load('../data/%s.npy' % data_name)

# Initial rectangle
if   data_name=='car1':  rect0 = np.array([170,130,290,250])
elif data_name=='car2':  rect0 = np.array([ 59,116,145,151])
elif data_name=='landing':rect0 = np.array([440,80,560,140])
else: raise ValueError

rects = [rect0]
fig, ax = plt.subplots()
numFrames = data.shape[2]

for i in range(numFrames-1):
    print("frame****************", i)
    It, It1 = data[:,:,i], data[:,:,i+1]
    rect = rects[-1]

    # run translation-only pyramid
    dx, dy = LucasKanadePyramid(It, It1, rect,
                                levels=3,    # try 3
                                scale=0.5)   # or tweak to 0.6/0.7
    print("p:", dx, dy)

    # update rectangle by translation
    newRect = rect + np.array([dx, dy, dx, dy])
    rects.append(newRect)

    # draw
    ax.add_patch(patches.Rectangle(
        (rect[0], rect[1]),
        rect[2]-rect[0]+1,
        rect[3]-rect[1]+1,
        linewidth=2, edgecolor='r', fill=False))
    plt.imshow(It1, cmap='gray')

    save_p = "../results/lk_pyramid/%s/frame%06d.jpg" % (data_name, i+1)
    mkdir_if_missing(save_p)
    plt.savefig(save_p)
    if do_display: plt.pause(0.01)
    ax.clear()

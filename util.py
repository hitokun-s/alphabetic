import logging
import os
import matplotlib.animation as animation
import numpy as np
from pylab import *


def checked(name):
    if not os.path.exists(name):
        os.mkdir(name)
    return name


def getFileLogger(name,filename=None,log_level=logging.INFO):
    if filename is None:
        filename = "%s.log" % name
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(name)

    file_handler = logging.FileHandler(filename, 'a+')
    file_handler.level = log_level
    logger.addHandler(file_handler)

    #test
    logger.debug('this is debug message')
    logger.info('this is info message')
    logger.warning('this is warning message')
    logger.error('this is error message')
    logger.critical('this is critical message')

    return logger


# image_arrays must be numpy image array
# you must install ffmpeg if you don't have one. Like this:
# http://www.faqforge.com/linux/how-to-install-ffmpeg-on-ubuntu-14-04/
# supported format type : gif, mp4
def animate(image_arrays, title="demo", size_inch=3, format="mp4"):
    print "Let's create awesome animation!"
    dpi = 100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    height, width = image_arrays[0].shape
    image_count = len(image_arrays)

    im = ax.imshow(rand(width, height), cmap='gray', interpolation='nearest')
    im.set_clim([0, 1])
    fig.set_size_inches([size_inch, size_inch]) # size of created video
    tight_layout()

    def update_img(n):
        im.set_data(image_arrays[n])
        return im

    # legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, frames=image_count, interval=30)
    filename = ""
    if format == "mp4":
        writer = animation.writers['ffmpeg'](fps=30)
        filename = '%s.mp4' % title
        ani.save(filename, writer=writer, dpi=dpi)
    elif format == "gif":
        filename = '%s.gif' % title
        ani.save(filename, writer='imagemagick')
    else:
        print "unsupported format type!:%s" % format
        return

    print "%s is created!" % filename
    return ani

# test to creating animation
if __name__ == "__main__":
    # animate([rand(100,100) for i in range(100)], title="test")
    animate([np.random.uniform(0, 1, (50, 50)).astype(np.float32) for i in range(100)], title="test")

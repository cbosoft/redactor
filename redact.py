from glob import glob

import numpy as np

from redactor.redact import Redactor


if __name__ == '__main__':
    redactor = Redactor()
    # imgs = glob('E:/Data/DF_DOE_LAMV_MeOH_26H/images/*.bmp')
    # imgs = glob('E:/Data/Kaggle/MARCO/extracted/*.jpg')
    imgs = glob('E:/Data/Kaggle/crystal-microscopics-images-and-annotations/extracted/*image.png')
    redactor.annot_and_redact(np.random.choice(imgs))

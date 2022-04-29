import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

_, im = cv.VideoCapture("ezgif.com-gif-maker.gif").read()

types = ['tiff', 'jpeg', 'png', 'jp2']
for t in types:
    cv.imwrite(f'im.{t}',im)

imgs = []
for t in types:
    imgs.append(cv.imread(f'im.{t}').astype(np.int32))

res = np.zeros((len(imgs),len(imgs)))
for i,ii in enumerate(imgs):
    for k, kk in enumerate(imgs):
        res[i,k] = np.sum(np.abs(kk-ii))/np.prod(ii.shape)

print(res)


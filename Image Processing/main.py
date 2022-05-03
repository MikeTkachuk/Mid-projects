import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# media types and frame retrieval
# - read from gif
# - show image formats reading and writing
# - show difference between formats

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

print('1.Lossless vs lossy compression')
printdf = pd.DataFrame(data=res, index=types, columns=types)
print(printdf,'\n\n')


# classical image operations
image = cv.imread('Kkxty.png')

# - color space conversion
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# - mask out all "red" items
red_low = np.uint8([170,20,20])
crack_high = np.uint8([179,255, 255])
crack_low = np.uint8([0,20, 20])

red_high = np.uint8([10,255,255])
mask = cv.inRange(image_hsv, red_low, crack_high) | cv.inRange(image_hsv, crack_low, red_high)

masked = cv.bitwise_and(image_hsv,image_hsv, mask=mask)

#plt.imshow(cv.cvtColor(masked, cv.COLOR_HSV2RGB))
#plt.show()

# - image linear transformation showcase
image = cv.imread('im.jpeg')
rows,cols,ch = image.shape

pts1 = np.float32([[50,50],[200,50],[50,200], [200,200]])
pts2 = np.float32([[10,100],[200,50],[100,250], [200,250]])
M = cv.getAffineTransform(pts1[:-1],pts2[:-1])
M1 = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpAffine(image,M,(cols,rows))
dst1 = cv.warpPerspective(image,M1,(cols,rows))
print(M)
print(M1)
plt.subplot(121),plt.imshow(dst),plt.title('Input')
plt.subplot(122),plt.imshow(dst1),plt.title('Output')
plt.show()

# - image kernel operations
# - image morphology algs https://en.wikipedia.org/wiki/Erosion_(morphology)
# - image derivatives https://towardsdatascience.com/image-derivative-8a07a4118550
# - canny
# -
image = cv.imread('im.jpeg')

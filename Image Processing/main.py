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

f, axes = plt.subplots(1,1,figsize=(12,7))
axes.imshow(cv.cvtColor(masked, cv.COLOR_HSV2RGB))

# - image linear transformation showcase
image = cv.imread('im.jpeg')
rows,cols,ch = image.shape

pts1 = np.float32([[50,50],[200,50],[50,200], [200,200]])
pts2 = np.float32([[60,60],[200,70],[50,200], [220,120]])
M = cv.getAffineTransform(pts1[:-1],pts2[:-1])
M1 = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpAffine(image,M,(cols,rows))
dst1 = cv.warpPerspective(image,M1,(cols,rows))

f, axes = plt.subplots(1,2,figsize=(12,7))
f.suptitle('Affine and Perspective transform')
axes = axes.flatten()

axes[0].imshow(dst), axes[0].scatter(*pts2[:-1].T),axes[0].title.set_text('Affine')
axes[1].imshow(dst1),axes[1].scatter(*pts2.T), axes[1].title.set_text('Perspective')


# - image kernel operations. both blurring and gaussian thresholding use kernels
# though differently

image = cv.imread('im.jpeg')
blurred = cv.GaussianBlur(cv.cvtColor(image, cv.COLOR_BGR2GRAY), (5,5), 1)
_, blurred = cv.threshold(blurred,127,255,cv.THRESH_BINARY)
# thresholding
thresholded = cv.adaptiveThreshold(cv.cvtColor(image, cv.COLOR_BGR2GRAY),255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,5,1)
f, axes = plt.subplots(2,2,figsize=(12,7))
axes = axes.flatten()
f.suptitle('Adaptive thresholding and binary thr. after blur is not the same!')

axes[0].imshow(image),\
axes[0].title.set_text('Input')
axes[1].imshow(blurred), axes[1].title.set_text('Output')
axes[2].imshow(thresholded), axes[2].title.set_text('Output')
axes[3].remove()


# - image morphology algs https://en.wikipedia.org/wiki/Erosion_(morphology)
# another use of kernels
image = cv.imread('im.jpeg')
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
print('Morphological elliptic kernel:')
print(kernel)
eroded = cv.erode(image,kernel)
dilated = cv.dilate(image, kernel)
grad = cv.morphologyEx(image, cv.MORPH_GRADIENT,kernel)
f, axes = plt.subplots(2,2,figsize=(12,7))
axes = axes.flatten()
f.suptitle('Morphology')

axes[0].imshow(image),\
axes[0].title.set_text('Input')
axes[1].imshow(eroded), axes[1].title.set_text('Erosion')
axes[2].imshow(dilated), axes[2].title.set_text('Dilation')
axes[3].imshow(grad), axes[3].title.set_text('Kernel gradient')


# - image kernel derivatives https://towardsdatascience.com/image-derivative-8a07a4118550
# kernels everywhere
image = cv.imread('im.jpeg')
sobel1 = cv.Sobel(image,cv.CV_32F,1,1,ksize=5)
sobel2 = cv.Sobel(sobel1,cv.CV_32F,1,1,ksize=5).astype(np.int32)
sobel3 = cv.Sobel(image, cv.CV_32F,2,2,ksize=5).astype(np.int32)
laplace = cv.Laplacian(image,cv.CV_32F).astype(np.int32)

f, axes = plt.subplots(2,2,figsize=(12,7))
axes = axes.flatten()
f.suptitle('Image derivatives')

axes[0].imshow(image),\
axes[0].title.set_text('Input')
axes[1].imshow(sobel2), axes[1].title.set_text('sobel 2nd consecutive')
axes[2].imshow(sobel3), axes[2].title.set_text('Sobel 2x2')
axes[3].imshow(laplace), axes[3].title.set_text('Laplace')


# - canny

image = cv.imread('im.jpeg')
canny = cv.Canny(image,20,50)
laplace = cv.Laplacian(image,cv.CV_32F)

f, axes = plt.subplots(2,2,figsize=(12,7))
axes = axes.flatten()
f.suptitle('Image derivatives')

axes[0].imshow(image),\
axes[0].title.set_text('Input')
axes[1].imshow(canny.astype(np.int32)), axes[1].title.set_text('Canny')
axes[2].imshow(laplace.astype(np.int32)), axes[2].title.set_text('Laplace')
axes[3].remove()

# -
plt.show()
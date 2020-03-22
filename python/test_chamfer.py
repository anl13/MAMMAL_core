import cv2 as cv
import numpy as np

testim = np.zeros((11,11), dtype = 'uint8')
testim[2:6,1:6] = 255
testim[3,3] = 0
print(testim)
dist = cv.distanceTransform(testim, cv.DIST_L2, 3)
print(dist)
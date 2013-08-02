import sys
import cv2
from helios import Flow

im1 = cv2.imread(sys.argv[1])
im2 = cv2.imread(sys.argv[2])
flo = Flow.load(sys.argv[3])

print im2.shape
flo = flo.resize((im2.shape[0], im2.shape[1]))

im2w = flo.warp(im2)
while True:
    cv2.imshow("im", im1)
    if cv2.waitKey(0) == 27:
        break
    cv2.imshow("im", im2w)
    if cv2.waitKey(0) == 27:
        break

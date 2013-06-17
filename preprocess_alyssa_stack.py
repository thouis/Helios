import sys
import os

import cv2
import fastremap
import clahe

if __name__ == "__main__":
    im = cv2.imread(sys.argv[1], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im = im[3249:47465, 5099:34750]

    scaledown = 16
    im = cv2.resize(im, (im.shape[1] // scaledown, im.shape[0] // scaledown))

    # reduce noise
    for i in range(2):
        im = cv2.medianBlur(im, 3)

    # equalize histogram
    clahe.clahe(im, im, 1.5)

    def halfstep(im):
        im = cv2.GaussianBlur(im, (0, 0), sigmaX=1.0)
        im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2))
        return im

    im = halfstep(im)
    cv2.imwrite(os.path.join(sys.argv[2], os.path.basename(sys.argv[1])),
                im)

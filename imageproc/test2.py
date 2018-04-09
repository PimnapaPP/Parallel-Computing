# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import multiprocessing as mp


# initial window position
posX = 20
posY = 20

# example for detect "coke" object function-------------------------------------------------
def detect4skal(hsv):
    global posX, posY

    # create mask image for detection of "coke" object
    # --create mask image for detection of red(upper) area in "coke" label
    rnb_red1 = createMaskImage(hsv, [161, 170], [50, 255], [50, 255])
    # --create mask image for detection of red(lower) area in "Skal" label
    rnb_red2 = createMaskImage(hsv, [0, 9], [50, 255], [50, 230])
    # --create mask image for detection of white area in "Skal" label
    msk_w = createMaskImage(hsv, [55, 90], [0, 50], [200, 255])
    # --create mask image for detection of "coke" object by adding msk_g to msk_b
    msk_b = createMaskImage(hsv, [110, 130], [0, 255], [50, 170])
    mask = cv2.addWeighted(rnb_red1, 0.1, rnb_red2, 0.1, 0)
    mask = cv2.addWeighted(msk_b, 1, mask, 0.1, 0)
    mask = cv2.addWeighted(msk_w, 0.1, mask, 1, 0)

    # show mask image
    cv2.imshow("only mask", mask)
    cv2.moveWindow("only mask", posX, posY + hsv.shape[0] + 40)

    # Salt and Pepper Noise Reduction using Opening process (Erosion next to Dilation) of mathematical morphology
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) # using 3x3 ellipse kernel
    cv2.imshow("adapt opening", mask)
    cv2.moveWindow("adapt opening", posX + hsv.shape[1] + 10, posY + hsv.shape[0] + 40)

    # fill hole in the object using Closing process (Dilation next to Erosion) of mathematical morphology
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))) # using 9x9 ellipse kernel
    cv2.imshow("adapt closing", mask)
    cv2.moveWindow("adapt closing", posX + hsv.shape[1]*2 + 25, posY + hsv.shape[0] + 40)

    # edge detection and extract contours
    edge = cv2.Canny(mask, 10, 80)
    im, contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # -- show extracted contours with gray line
    cv2.drawContours(mask, contours, -1, (128, 128, 128), 2)

    # show bounding box of extracted objects using contours
    for i in range(len(contours)):
        # -- get information of bounding rect of each contours
        x, y, w, h = cv2.boundingRect(contours[i])
        # -- decide "Skal" object using aspect ratio of bounding area
        if w*1.8<h and h<w*2.5: # -- is "Skal"
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            strSize = cv2.getTextSize("Skal(full of)", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x, y-strSize[1]), (x+strSize[0], y), (0, 0, 255), -1)
            cv2.putText(img, "Skal(full of)", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        else: # -- is not "Skal"
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("target image", img)

    return mask
# create sample mask image window function-------------------------------------------------
def createMaskImage(hsv, hue, sat, val):
    imh, imw, channels = hsv.shape  # get image size and the number of channels
    mask = np.zeros((imh, imw, channels), np.uint8) # initialize hsv gradation image with 0

    if isinstance(hue, list): # if hue argument is pair value enclosed in []
        hmin = hue[0]
        hmax = hue[1]
    else:                     # if hue argument is single value
        hmin = hue
        hmax = hue

    if isinstance(sat, list): # if sat argument is pair value enclosed in []
        smin = sat[0]
        smax = sat[1]
    else:                     # if sat argument is single value
        smin = sat
        smax = sat

    if isinstance(val, list): #  val argument is pair value enclosed in []
        vmin = val[0]
        vmax = val[1]
    else:                    # if val argument is single value
        vmin = int(val)
        vmax = int(val)

    return cv2.inRange(hsv, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))

# main function-----------------------------------------------------------------------------
def main():
    global img, cache
    global posX, posY
    A = os.listdir('./')
    matchers = ['png','jpg','jpeg']
    B = [s for s in A if any(xs in s for xs in matchers)]
    print(B)
    # read image
    for i in range(len(B)):
        img = cv2.imread("./"+B[i])
        imh, imw, channels = img.shape  # get image size and the number of channels

        # pre-processing area ---------------------
        #img = cv2.GaussianBlur(img, (5, 5), 5)


        #-------------------------------------------
        cache = img.copy()

        # display image
        cv2.imshow("target image", img)
        cv2.moveWindow("target image", posX, posY)

        # calc window position --------
        X = posX + imw + 10
        Y = posY
        # ------------------------------

        # convert to HSV (Hue, Saturation, Value(Brightness))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        cv2.imshow("detect skal", detect4skal(hsv))
        cv2.moveWindow("detect skal", X, Y)

        # keep all windows until "ESC" button is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

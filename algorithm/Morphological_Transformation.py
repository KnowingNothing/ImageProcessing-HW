import cv2
import numpy as np


def MyErosion(img, kernel):
    result = cv2.erode(img, kernel)
    return result

def MyDilation(img, kernel):
    result = cv2.dilate(img, kernel)
    return result

def MyOpening(img, kernel):
	return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def MyClosing(img, kernel):
	return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


if __name__ == '__main__':
    pass
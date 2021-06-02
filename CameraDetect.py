import Detector
import cv2

cam = cv2.VideoCapture(0)
while True:
    re, frame = cam.read()
    newFrame = Detector.detect(frame)
    cv2.imshow('Camera', newFrame)
    cv2.waitKey(20)
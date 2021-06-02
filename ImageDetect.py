import Detector2
import cv2

imgTest = cv2.imread('ImgTest.jpg')
newImg = Detector2.detect(imgTest)
cv2.imshow('Result', newImg)
cv2.waitKey(5000)
cv2.destroyAllWindows()
import cv2
import numpy as np

imgTrain = cv2.imread('ImgTrain.png')


def createDetector():
    detector = cv2.ORB_create(nfeatures=2000)
    return detector

def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = createDetector()
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs, img.shape[:2][::-1]

def detectFeatures(img, train_features):
    train_kps, train_descs, shape = train_features
    # get features from input image
    kps, descs, _ = getFeatures(img)
    # check if keypoints are extracted
    if not kps:
        return None
    # now we need to find matching keypoints in two sets of descriptors (from sample image, and from current image)
    # knnMatch uses k-nearest neighbors algorithm for that
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(train_descs, descs, k=2)
    good = []
    # apply ratio test to matches of each keypoint
    # idea is if train KP have a matching KP on image, it will be much closer than next closest non-matching KP,
    # otherwise, all KPs will be almost equally far
    for m, n in matches:
        if m.distance < 1.2 * n.distance:
            good.append([m])
    # stop if we didn't find enough matching keypoints
    if len(good) < 0.1 * len(train_kps):
        return None, 0
    # estimate a transformation matrix which maps keypoints from train image coordinates to sample image
    src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)


    m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    m2, mask = cv2.findHomography(dst_pts, src_pts,  cv2.RANSAC, 5.0)
    if m is not None:
        # apply perspective transform to train image corners to get a bounding box coordinates on a sample image
        scene_points = cv2.perspectiveTransform(np.float32([(0, 0), (0, shape[0] - 1), (shape[1] - 1, shape[0] - 1), (shape[1] - 1, 0)]).reshape(-1, 1, 2), m)
        rect = cv2.minAreaRect(scene_points)
        # check resulting rect ratio knowing we have almost square train image
        if rect[1][1] > 0 and 0.8 < (rect[1][0] / rect[1][1]) < 1.2:
            return rect, m2
    return None, 0


def detect(test):
    region, m = detectFeatures(test, train_features)
    if region is not None:
        
        box = cv2.boxPoints(region)
        box = np.int0(box)
        
        
        w = getTrainImgSize()[0]
        h = getTrainImgSize()[1]

        # lấy ra hình ảnh logo được phát hiện và xoay nó về góc nhìn chuẩn
        test1 = cv2.warpPerspective(test, m, (w, h))
        # vẽ đường bao quanh logo được phát hiện
        cv2.drawContours(test, [box], 0, (255, 0, 0), 2)

        test1 = cv2.resize(test1, dsize=(int(w/5), int(h/5)))
        test[0:int(h/5), 0:int(w/5)] = test1
    else:
        cv2.putText(test, "Khong phat hien duoc logo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return test

def getTrainImgSize():
    return imgTrain.shape


train_features = getFeatures(imgTrain)


import cv2
import numpy as np
import os 


imgTrain = cv2.imread('ImgTrain.png')


def createDetector():
    detector = cv2.ORB_create(nfeatures=2000)
    return detector

def getFeatures(img):
    # tìm các keypoint, descriptor trong hình ảnh bằng ORB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = createDetector()
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs, img.shape[:2][::-1]

def detectFeatures(img, train_features):
    train_kps, train_descs, shape = train_features
    # Lấy các keypoint và descriptor từ ảnh đầu vào
    kps, descs, _ = getFeatures(img)

    # kiểm tra xem có tìm ra được keypoint nào không
    if not kps:
        return None


    # Sử dụng BFMatcher (brute force matcher) để tìm ra 2 keypoint giống nhất với từng keypoint có trong ảnh đầu vào.
    # Độ đo được sử dụng là Hamming distance
    # Hamming distance là đại lượng để so sánh 2 chuỗi bit, nó là tổng số bit khác nhau ở từng vị trí của mỗi chuỗi.
    # knnMatch sử dụng k-nearest neighbors để tìm ra 2 keypoint gần giống nhất


    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(train_descs, descs, k=2)
    good = []


    # Sử dụng ratio test để tìm ra các keypoint tương ứng
    # Nếu một keypoint ở ảnh train tương ứng với một keypoint khác ở ảnh cần tìm thì nó sẽ gần hơn nhiều so với 
    # keypoint khác nó nhưng gần hơn nhất.
    # => so sánh độ khác nhau giữa keypoint của ảnh train với 2 keypoint tìm được, độ khác biệt giữa hai độ khác nhau
    #  này là đủ lớn thì có thể sử dụng keypoint đó có tính toán, nếu 2 không thì keypoint đó không được sử dụng.
    for m, n in matches:
        if m.distance < 1.2 * n.distance:
            good.append([m])


    # Dừng lại nếu ta không tìm được đủ số keypoint cần thiết
    
    if len(good) < 0.1 * len(train_kps):
        return None, 0

    # Tìm ra ma trận chuyển đổi để ánh xạ giữa các toạ đổ của các keypoint từ ảnh train và ảnh đầu vào
    src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)

    m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    m2, mask = cv2.findHomography(dst_pts, src_pts,  cv2.RANSAC, 5.0)


    if m is not None:
        # Sử dụng ma trận chuyển đổi vào các góc của ảnh train để tìm ra toạ độ hình chữ nhật viền xung quanh ảnh đầu vào
        scene_points = cv2.perspectiveTransform(np.float32([(0, 0), (0, shape[0] - 1), (shape[1] - 1, shape[0] - 1), (shape[1] - 1, 0)]).reshape(-1, 1, 2), m)
        rect = cv2.minAreaRect(scene_points)

        # Vì biết trước ảnh train có dạng hình gần vuông nên kiểm tra nên hình chữ nhât tìm được cũng phải như vậy
        if rect[1][1] > 0 and 0.8 < (rect[1][0] / rect[1][1]) < 1.2:
            return rect, m2

    return None, 0


def detect(test):
    region, m = detectFeatures(test, train_features)
    if region is not None:
        
        box = cv2.boxPoints(region)
        box = np.int0(box)
        print(box)
        
        w = getTrainImgSize()[0]
        h = getTrainImgSize()[1]

        # lấy ra hình ảnh logo được phát hiện và xoay nó về góc nhìn chuẩn
        test1 = cv2.warpPerspective(test, m, (w, h))
        # vẽ đường bao quanh logo được phát hiện
        cv2.drawContours(test, [box], 0, (0, 255, 0), 2)

        test1 = cv2.resize(test1, dsize=(int(w/5), int(h/5)))
        test[0:int(h/5), 0:int(w/5)] = test1
    else:
        cv2.putText(test, "Khong phat hien duoc logo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return test

def getTrainImgSize():
    return imgTrain.shape


train_features = getFeatures(imgTrain)


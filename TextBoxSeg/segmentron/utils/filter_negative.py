import cv2
import os
# import cv2
import random
import numpy as np

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def get_negative(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=1)
    cv2.imwrite("/mnt/lustre/xieenze/wuweijia/CVPR_SemiText/SemiText/TextBoxSeg/workdirs/show/edge.jpg",
                canny)

    vis_3 = np.zeros(img.shape[:2])
    blurred = cv2.blur(canny, (9, 9))

    _, thresh = cv2.threshold(blurred, 5, 1, cv2.THRESH_BINARY)

    thresh = np.array(1 - thresh)
    # thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    dist = cv2.distanceTransform(thresh.copy().astype(np.uint8), cv2.DIST_L2, 5)

    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imwrite("/mnt/lustre/xieenze/wuweijia/CVPR_SemiText/SemiText/TextBoxSeg/workdirs/show/distance.jpg",
                cvt2HeatmapImg(dist))
    distance_map = np.array(dist > 0.1) * 1
    cv2.imwrite("/mnt/lustre/xieenze/wuweijia/CVPR_SemiText/SemiText/TextBoxSeg/workdirs/show/final_.jpg",
                distance_map*255)
    # distance_map = cv2.threshold(dist, 0.5, 1, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(distance_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    for idx, cnt in enumerate(cnts):
        # cnt = cnts[idx]
        if idx >= 4:
            break
        if cv2.contourArea(cnt) < 2000 and idx > 0:
            break
        vis_3 = cv2.fillPoly(vis_3, [cnt], 1)

    vis_3 = vis_3*distance_map
    return vis_3

def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect

#
image_path = "/mnt/lustre/xieenze/wuweijia/CVPR_SemiText/SemiText/TextBoxSeg/workdirs/show/IMG_0003.jpg"

# image_list = os.listdir(image_path)
# idx = random.randint(0,1000)
# image_path_ = os.path.join(image_path,image_list[idx])
img = cv2.imread(image_path)
background = get_negative(img)
# cv2.imwrite("/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet_self_training/workdirs/icdar15/image_result_1.jpg",boxing)
# vis_1 = img.copy()
# vis_2 = img.copy()
# vis_3 = np.zeros_like(img)
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# gray = cv2.GaussianBlur(gray,(3,3),0)
# canny = cv2.Canny(gray, 50, 150)
#
# blurred = cv2.blur(canny, (9, 9))
#
# _, thresh = cv2.threshold(blurred, 5, 1, cv2.THRESH_BINARY)
# # print(thresh.max())
# # cnts, _ = cv2.findContours( (1-thresh).copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
# # 先找出轮廓点
# #### distance transform
# thresh = np.array(1-thresh)
# # thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
#
# dist = cv2.distanceTransform(thresh.copy().astype(np.uint8),  cv2.DIST_L2, 5)
# cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
#
# distance_map = np.array(dist>0.3)*1
# # distance_map = cv2.threshold(dist, 0.5, 1, cv2.THRESH_BINARY)
# cnts, _ = cv2.findContours( distance_map.astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnts.sort(key=lambda x:cv2.contourArea(x), reverse=True)
#
# for idx,cnt in enumerate(cnts):
#     # cnt = cnts[idx]
#     if idx>=3:
#         break
#     if cv2.contourArea(cnt)<2500 and idx>0:
#         break
#     vis_3 = cv2.fillPoly(vis_3,[cnt],255)
# # rect = order_points(c.reshape(c.shape[0], 2))
# # print(rect)
# print(vis_3.shape)
# # cv2.fillPoly(vis_3,np.array([rect]).astype(np.int32),255)
# # 合并图片
# canny =  np.expand_dims(canny, axis=2)
# canny = np.concatenate((canny, canny, canny), axis=-1)
#
# # dist =  np.expand_dims(dist, axis=2)
# dist =  np.expand_dims(dist, axis=2)
# dist = np.concatenate((dist, dist, dist), axis=-1)
# print(canny.shape)
# boxing_list = [img, canny, dist*255, vis_3]
# boxing = np.concatenate(boxing_list, axis=1)
#
# cv2.imwrite("/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet_self_training/workdirs/icdar15/image_result_1.jpg",boxing)
#coding=UTF-8
import cv2
pathSource=".\\myImage\\lzh001.png"
pathTarget=".\\predictions\\lzh001.png"
source = cv2.imread(pathSource)
target = cv2.imread(pathTarget, cv2.IMREAD_GRAYSCALE)

heatMap_pre = cv2.applyColorMap(target, cv2.COLORMAP_JET)

overlay = source.copy()

height = target.shape[0]
weight = target.shape[1]
alpha = 0.7

predictionMap = source.copy()
cv2.rectangle(overlay, (0, 0), (source.shape[1], source.shape[0]), (105, 105, 105), -1)  # 设置蓝色为热度图基本色
# cv2.addWeighted(overlay,alpha,source,1-alpha,0,source)
cv2.addWeighted(heatMap_pre, alpha, source, 1 - alpha, 15, predictionMap)

cv2.namedWindow("aa",cv2.WINDOW_NORMAL)
cv2.imshow("aa",predictionMap)
cv2.waitKey(0)

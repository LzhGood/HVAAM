#coding=UTF-8
import cv2
import os

index=111
cate="vehicle"
source=cv2.imread("E:/Dataset/MyData/original500Correction_classified/val/Source/"+cate+"/"+str(index)+".png")
target=cv2.imread("E:/Dataset/MyData/original500Correction_classified/val/Prediction/SAM-With-Strategy-1.6771/"+cate+"/"+str(index)+".png",cv2.IMREAD_GRAYSCALE)
groundTruth=cv2.imread("E:/Dataset/MyData/original500Correction_classified/val/SaliencyMap/"+cate+"/"+str(index)+".png",cv2.IMREAD_GRAYSCALE)



heatMap_pre = cv2.applyColorMap(target, cv2.COLORMAP_JET)
heatMap_tru = cv2.applyColorMap(groundTruth, cv2.COLORMAP_JET)

overlay=source.copy()

height=target.shape[0]
weight=target.shape[1]
alpha=0.7#0.7

predictionMap=source.copy()
cv2.rectangle(overlay, (0, 0), (source.shape[1], source.shape[0]), (105, 105, 105), -1) # 设置蓝色为热度图基本色
# cv2.addWeighted(overlay,alpha,source,1-alpha,0,source)
cv2.addWeighted(heatMap_pre,alpha,source,1-alpha,15,predictionMap)

trueMap=source.copy()
cv2.rectangle(overlay, (0, 0), (source.shape[1], source.shape[0]), (105, 105, 105), -1) # 设置蓝色为热度图基本色
# cv2.addWeighted(overlay,alpha,source,1-alpha,0,source)
cv2.addWeighted(heatMap_tru,alpha,source,1-alpha,15,trueMap)

cv2.imshow("Source", source)
cv2.imshow("Prediction", predictionMap)
cv2.imshow("GroundTruth", trueMap)
key=cv2.waitKey(0)


print (key)
if(key == 83):#S
    print ("Save")
    cv2.imwrite("./need/"+cate+"/"+str(index)+"_prediction.png",predictionMap,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    cv2.imwrite("./need/"+cate+"/"+str(index)+"_source.png",source,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    cv2.imwrite("./need/"+cate+"/"+str(index) + "_Ground.png", trueMap, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
else:
    exit(1)

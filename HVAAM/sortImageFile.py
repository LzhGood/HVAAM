#coding=UTF-8
import os
import cv2
path="C:\\Users\media\\"
outdir="E:\Dataset\Interface\Image\\"
imageList=os.listdir(path)
print(len(imageList))
# exit(0)
for i in range(len(imageList)):
    print(path+imageList[i]+"\n")
    # exit(0)
    image=cv2.imread(path+imageList[i],cv2.IMREAD_COLOR)
    # r,g,b=cv2.split(image)
    # image=cv2.merge([r,g,b])
    # # cv2.imshow("aaa",image)
    # # cv2.waitKey(0)
    # # exit(0)

    cv2.imwrite(outdir+str(i+271)+'.png',image,[int(cv2.IMWRITE_PNG_COMPRESSION),0])

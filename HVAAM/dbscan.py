#coding=UTF-8
from natsort import natsorted
import os
from PIL import Image
import numpy as np
import Augmentor
import cv2
from sklearn.cluster import DBSCAN

def dbscan(img):#img:numpy array
    input=[]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                b=[j,i]
                input.append(b)

    cls = DBSCAN(eps=4, min_samples=1).fit(input)

    num_cls=len(set(cls.labels_))
    set_cls=set(cls.labels_)
    input=np.asarray(input)

    listPoint = np.zeros([num_cls, 2], dtype=float)
    listNum = np.zeros([num_cls], dtype=float)
    for i in range(input.shape[0]):
        for n in set_cls:
            if (cls.labels_[i] == n):
                if (n != -1):
                    listPoint[n, :1] += input[i, 0]
                    listPoint[n, 1:] += input[i, 1]
                    listNum[n] += 1
                break

    for i in range(listPoint.shape[0]):
        listPoint[i, :1] /= listNum[i]
        listPoint[i, 1:] /= listNum[i]

    result=np.zeros([480,640],dtype=np.uint8)

    for i in range(listPoint.shape[0]):
        result[int(listPoint[i,1]+0.5)][int(listPoint[i,0]+0.5)]=255

    return result


cv2.namedWindow("AAA",cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("AAA",100,100)
cv2.namedWindow("BBB",cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("BBB",120,120)
cv2.namedWindow("CCC",cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("CCC",140,140)

fixationPath="E:\\Dataset\\train\\FixationMap\\"
saliencyPath="E:\\Dataset\\train\\SaliencyMap\\"
sourcePath="E:\\Dataset\\train\\Source\\"

fixationPathOut="E:\\Datasettrain\\FixationMap\\"
saliencyPathOut="E:\\Datasettrain\\SaliencyMap\\"
sourcePathOut="E:\\Dataset\\train\\Source\\"

fList = [fixationPath + f for f in os.listdir(fixationPath) if f.endswith(('.jpg', '.jpeg', '.png'))]
sList=[saliencyPath + f for f in os.listdir(saliencyPath) if f.endswith(('.jpg', '.jpeg', '.png'))]
List=[sourcePath + f for f in os.listdir(sourcePath) if f.endswith(('.jpg', '.jpeg', '.png'))]

# for i in range(len(fList)):
#     tmp=cv2.imread(fList[i],cv2.IMREAD_GRAYSCALE)
#     tmp=tmp*255
#     cv2.imwrite(fList[i], tmp, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
#
# exit(1)

fixationSortedList=natsorted(fList)
saliencySortedList=natsorted(sList)
sourceSortedList=natsorted(List)

collatedList=list(zip(sourceSortedList,saliencySortedList,fixationSortedList))

num=440#281
collatedList_=collatedList[0:num]

images=[[np.asarray(Image.open(y)) for y in x] for x in collatedList_]
p=Augmentor.DataPipeline(images)

# p.shear(1,23,23)
# p.rotate(1,15,15)
p.random_distortion(1,9,9,10)

p.skew_tilt(1,0.7)
p.skew_corner(1,0.7)
# p.random_erasing(1,0.5)
# p.shear(1,20,20)
# p.crop_random(1,0.7)
# p.resize(1,640,480)


g=p.generator(1)

bias=2880
for i in range(0,num):
    tmp = next(g)
    img = cv2.cvtColor(np.asarray(tmp[0][0]), cv2.COLOR_RGB2BGR)

    # cv2.imshow("AAA",img)
    # cv2.imshow("BBB",tmp[0][1])
    # cv2.imshow("CCC",tmp[0][2])
    # cv2.waitKey(0)
    # continue

    # exit(1)

    cv2.imwrite(sourcePath+str(bias+i)+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    cv2.imwrite(saliencyPath+str(bias+i)+".png", np.asarray(tmp[0][1]).astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    cv2.imwrite(fixationPath+str(bias+i)+".png", dbscan(np.asarray(tmp[0][2]).astype(np.uint8)), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


exit(1)




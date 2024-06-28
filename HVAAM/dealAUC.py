#coding=UTF-8
import os
import numpy as np


#aera under curve
def auc(x,y):#x,y input list
    # tprList=np.array(tprList)
    # fprList=np.array(fprList)
    s=[]
    for i in range(len(x)):
        s.append([x[i],y[i]])

    s=np.array(s).astype(np.float32)

    s=s[np.argsort(s[:,0])]

    x_=[]
    y_=[]
    for i in range(len(x)):
        x_.append(s[i][0])
        y_.append(s[i][1])

    return np.trapz(y_,x_)

rootPath="E:/Dataset/val/Prediction/"
pathList=[]
classList=['panel/ROC/','vehicle/ROC/','web/ROC/']


for p in pathList:
    mean=0
    for c in classList:
        sum = 0
        files = os.listdir(rootPath+p+c)
        for f in files:
            inputData = np.loadtxt(rootPath+p+c+f, unpack=True, delimiter=' ', usecols=[0, 1]).tolist()
            tmp=auc(inputData[0],inputData[1])
            if(f != "ROC_Mean.txt"):
                sum+=tmp
            txtName=open(rootPath+p+c+f[:-4] + "-AUC-"+str(tmp)+".txt","w+")
            txtName.write(str(tmp))
            txtName.close()

        mean+=sum/(len(files)-1)
        txtName = open(rootPath + p + c + "Real-Mean-AUC-" + str(sum/(len(files)-1)) + ".txt", "w+")
        txtName.write(str(sum/(len(files)-1)))
        txtName.close()

    txtName = open(rootPath + p + "Mean-AUC-" + str(mean/3) + ".txt", "w+")
    txtName.write(str(mean/3))
    txtName.close()



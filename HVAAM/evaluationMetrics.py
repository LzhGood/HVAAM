#coding=UTF-8
import keras.backend as K
from config import *
import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

#计算皮尔逊相关度：
def pearson(saliencyG,saliencyP):
    p=saliencyG
    q=saliencyP
    #分别求p，q的和
    n = len(p)
    sumx = sum([p[i] for i in range(n)])
    sumy = sum([q[i] for i in range(n)])
    #分别求出p，q的平方和
    sumxsq = sum([p[i]**2 for i in range(n)])
    sumysq = sum([q[i]**2 for i in range(n)])
    #求出p，q的乘积和
    sumxy = sum([p[i]*q[i] for i in range(n)])
    # print sumxy
    #求出pearson相关系数
    up = sumxy - sumx*sumy/n
    down = ((sumxsq - pow(sumx,2)/n)*(sumysq - pow(sumy,2)/n))**.5
    #若down为零则不能计算，return 0
    if down == 0 :
        return 0
    r = up/down
    return r

def kl(saliencyG,saliencyP):
    np_g=np.array(saliencyG)
    np_p=np.array(saliencyP)

    np_g/=np_g.sum()
    np_p/=np_p.sum()
    for i in range(np_p.shape[0]):
        np_p[i]=np_g[i]*math.log((np_g[i]/(np_p[i] + K.epsilon()))+K.epsilon())#K.epsilon() = 1^(-7)
    return np_p.sum()


def sim(saliencyG,saliencyP):
    np_g=np.array(saliencyG)
    np_p=np.array(saliencyP)

    np_g/=np_g.sum()
    np_p/=np_p.sum()

    for i in range(np_p.shape[0]):
        np_p[i]=min(np_p[i],np_g[i])

    return np_p.sum()


def nss(fixationG,saliencyP):
    np_g=np.array(fixationG)
    np_p=np.array(saliencyP)
    #np_g/=255.0
    np_p/=255.0

    std_p=np.std(np_p,ddof=1)
    mean_p=np_p.mean()

    N = np_g.sum()

    result=sum(((np_p[i]-mean_p)/std_p*np_g[i]) for i in range(len(saliencyP))) /N
    return result

#roc曲线
def roc(fixationG,saliencyP):
    np_g=np.array(fixationG)
    np_g=np_g*255;
    np_p=np.array(saliencyP)

    tprList=[]
    fprList=[]
    for threshold in range(0,256,1):
        tp=0.0
        fp=0.0
        tn=0.0
        fn=0.0
        for i in range(np_p.shape[0]):
            if np_p[i] >= threshold:
                if  np_g[i] > 100:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if np_g[i] > 100:
                    fn=fn+1
                else:
                    tn=tn+1

        tprList.append(tp/(tp+fn))
        fprList.append(fp/(fp+tn))
    return [tprList,fprList]#y,x




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

def auc_embed_roc(fixationG,saliencyP):
    result=roc(fixationG,saliencyP)
    return auc(result[1],result[0])


groundTruthPath="D:/humanComputerInteraction/TruthSali/Sa/"
groundFixtionPath="D:/humanComputerInteraction/TruthSali/Fi/"
predPath="D:/humanComputerInteraction/PreSali/"

gList = [groundTruthPath + f for f in os.listdir(groundTruthPath) if f.endswith(('.jpg', '.jpeg', '.png'))]
gFList = [groundFixtionPath + f for f in os.listdir(groundFixtionPath) if f.endswith(('.jpg', '.jpeg', '.png'))]
pList=[predPath + f for f in os.listdir(predPath) if f.endswith(('.jpg', '.jpeg', '.png'))]

gList.sort()
gFList.sort()
pList.sort()

suma=0
imsG = np.zeros((1,shape_r_out, shape_c_out))
imsFG = np.zeros((1,shape_r_out, shape_c_out))
imsP = np.zeros((1,shape_r_out, shape_c_out))

G=np.zeros((shape_r_out, shape_c_out))
GF=np.zeros((shape_r_out, shape_c_out))
P=np.zeros((shape_r_out, shape_c_out))

G_=[]
GF_=[]
P_=[]

for i in range(len(gList)):
    gMap=cv2.imread(gList[i],cv2.IMREAD_GRAYSCALE)
    gFMap = cv2.imread(gFList[i], cv2.IMREAD_GRAYSCALE)
    pMap=cv2.imread(pList[i],cv2.IMREAD_GRAYSCALE)
    G=gMap.astype(np.float32)
    GF = gFMap.astype(np.float32)
    P=pMap.astype(np.float32)

    # cv2.imshow("G",gMap)
    # cv2.imshow("P",pMap)
    # cv2.waitKey(0)
    # print(str(i))
    # print(gList[i])
    for r in range(0,480):
        for c in range(0,640):
            G_.append(G[r,c])
            GF_.append(GF[r, c])
            P_.append(P[r,c])

    print(gList[i])
    result=roc(GF_,P_)
    y=result[0]
    x=result[1]
    print ("auc:",auc(x,y))

    print("KL:",1.3-kl(G_,P_))
    print("SIM:",sim(G_,P_))
    print("CC:",pearson(G_, P_))
    print("NSS:",nss(GF_,P_))
    #plt.plot(x,y)
    #plt.show()
    # exit(1)

    # a=pearson(G_, P_)
    # a=nss(G_,P_)
    # a=sim(G_,P_)
    # a=kl(G_,P_)
    #a=auc(x,y)
    #print(a)
    #suma=suma + a
    # print(str(i)+" "+str(a))
    G_=[]
    P_=[]
    result=[]
    # print("*****************")

#print(suma/len(gList))



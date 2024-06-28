#coding=UTF-8
#模块调用
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN

def filterData(name,nameIndex,rangeBegin,rangeEnd):
    name_=name+"_"+str(nameIndex)
    for index in range(rangeBegin, rangeEnd):
        print("Processing "+name_+" File " + str(index))
        X = np.loadtxt("./Completed/"+name_+"/" + str(index) + ".txt", unpack=False, delimiter=' ', usecols=[0, 1, 2])


        ind=X.shape[0]-1
        for n in range(0,X.shape[0]):
            if X[n,2]-X[0,2] > 2000:
                ind=n
                break
        for n in range(X.shape[0]-1,ind-1,-1):
            X=np.delete(X,n,axis=0)

        X = X[:, :2]

        # print(X)
        # exit(1)

        X[:, 0] -= 329
        X[:, 0] *= 0.5
        X[:, 1] -= 112
        X[:, 1] *= 0.5

        cls = DBSCAN(eps=10, min_samples=8).fit(X)#10,8
        n_clusters = len(set(cls.labels_))
        set_clusters = set(cls.labels_)

        isHaveN = False
        for tmp in set_clusters:
            if (tmp == -1):
                isHaveN = True

        if (isHaveN == True):
            listPoint = np.zeros([n_clusters - 1, 2], dtype=float)
            listNum = np.zeros([n_clusters - 1], dtype=float)
        else:
            listPoint = np.zeros([n_clusters, 2], dtype=float)
            listNum = np.zeros([n_clusters], dtype=float)

        for i in range(X.shape[0]):
            for n in set_clusters:
                if (cls.labels_[i] == n):
                    if (n != -1):
                        listPoint[n, :1] += X[i, 0]
                        listPoint[n, 1:] += X[i, 1]
                        listNum[n] += 1
                    break

        for i in range(listPoint.shape[0]):
            listPoint[i, :1] /= listNum[i]
            listPoint[i, 1:] /= listNum[i]

        if not os.path.exists("./Completed_output/"+name+"_output/"):
            os.makedirs("./Completed_output/"+name+"_output/")
        fileName = open("./Completed_output/"+name+"_output/" + str(index) + ".txt", 'w+')
        for i in range(listPoint.shape[0]):
            fileName.write(str(int(listPoint[i, :1] + 0.5)))  # x
            fileName.write(' ')
            fileName.write(str(int(listPoint[i, 1:] + 0.5)))  # y
            fileName.write("\n")


nameList=[]

for name in nameList:
    for nameIndex in range(1,4):
        if nameIndex == 1:
            filterData(name,nameIndex,0,160)
        elif nameIndex == 2:
            filterData(name,nameIndex,160,320)
        elif nameIndex == 3:
            filterData(name,nameIndex,320,500)

exit(1)

markers = ['^', 'x', 'o', '*', '+','<','>','H','1','2','3']
for i in range(n_clusters):
    my_members = cls.labels_ == i
    plt.scatter(X[my_members, 0], X[my_members, 1], s=60, marker=markers[i], c='blue', alpha=0.1)
plt.title('dbscan')
plt.show()


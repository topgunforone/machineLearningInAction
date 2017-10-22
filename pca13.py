#coding:utf-8
from numpy import *
import matplotlib.pyplot  as plt
def loadDataSet(filename,delim='\t'):
    fr =open(filename)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[map(float,line) for line in stringArr]
    return mat(datArr)


def pca(dataMat,topNfeat=999999):
    #均值化
    meanVals=mean(dataMat,0)
    meanRemoved=dataMat-meanVals
    covMat=cov(meanRemoved,rowvar=0)#rowvar=0表示行是观测向量
    eigVals,eigVects=linalg.eig(mat(covMat))
    eigValInd=argsort(eigVals)
    eigValInd=eigValInd[-1:-topNfeat-1:-1]#超出界限就到0，不会报错
    redEigVects=eigVects[:,eigValInd]
    lowDataMat=meanRemoved*redEigVects
    varRecord=cov(lowDataMat,rowvar=0)
    reconMat=(lowDataMat*redEigVects.T)+meanVals
    # return lowDataMat,reconMat,redEigVects
    return  meanRemoved,meanVals,redEigVects8

if __name__=='__main__':
    datMat=mat(loadDataSet('testSet13.txt'))
    meanRemoved, meanVals, redEigVects=pca(datMat,1)#recondMats是用原始的坐标重构降维后的数据。表现为垂直轴方向无数据
    # print lowDMat,reconMat
    # print varRecord
    # plt.figure()
    # plt.scatter(datMat[:,0],datMat[:,1])
    # plt.scatter(reconMat[:,0],reconMat[:,1])
    # plt.show()
    print redEigVects






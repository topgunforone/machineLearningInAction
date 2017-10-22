#coding:utf-8
from numpy import *


def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=map(float,curLine)
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))


#generate k centroids
def randCent(dataSet,k):
    n=shape(dataSet)[1]#
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*random.random_sample((k,1))
    return  centroids



def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    '''
    k均值聚类
    :param dataSet:
    :param k:
    :param distMeas:
    :param createCent:
    :return: 返回每个类别的中心点坐标和 各个样本到各自属于中心的聚类(可自定义)
    '''
    m=shape(dataSet)[0]
    clusterAssement=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)#随机生成初始点
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):#样本个数
            minDist=inf;minIndex=-1
            for j in range(k):
                distJI=distMeas(centroids[j,:],dataSet[i,:])#计算样本到每个类别的距离
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if clusterAssement[i,0]!=minIndex:clusterChanged=True
            clusterAssement[i,:]=minIndex,minDist**2  #类别及距离
        print centroids
        #每次所有的样本点更新完之后，更新聚类中心
        for cent in range(k):
            ptsInClust=dataSet[nonzero(clusterAssement[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,0)
    return  centroids,clusterAssement


#二分聚类

def biKmeans(dataSet,k,distMeas=distEclud):
    m=shape(dataSet)[0]
    #create initial centroid
    clusterAssement=mat(zeros((m,2)))#第一列类别，第二列距离
    centroid0=mean(dataSet,0).tolist()[0]#第一次的中心是所有点的均值
    centList=[centroid0]#均值点 样本点的顺序就是他们的类别
    for j in range(m): #每个样本到原始样本的距离
        clusterAssement[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2#第0类并且到点的距离
    while len(centList)<k:#当聚类的类别个数不到K时
        lowestSSE=inf
        for i in range(len(centList)):#每个类别都划分一遍，筛选出最好结果
            ptsInCurrCluster=dataSet[nonzero(clusterAssement[:,0].A==i)[0],:]#属于第i类的数据集合
            centroidMat,splitClustAss=kMeans(ptsInCurrCluster,2,distMeas) #kmean为两类
            sseSplit=sum(splitClustAss[:,1])#分类之后，各点到各自的聚类中心距离和
            sseNoSplit=sum(clusterAssement[nonzero(clusterAssement[:,0].A!=i)[0],1])#剩余类别，即当前假设第i类划分，则剩余其他类为划分
            # #sseSplit+sseNoSpli 表示的是一部分划分，加上其他未划分的。可以简化为划分前和后的误差比较
            # ssePostSplit = sum(splitClustAss[:, 1])  # 分类之后，各点到各自的聚类中心距离和
            # ssePreSplit=clusterAssement[nonzero(clusterAssement[:,0].A==i)[0]][1]
            #----------
            if (sseSplit+sseNoSplit)<lowestSSE:#寻找最优划分，
                bestCentToSplit=i#原类别
                bestNewCents=centroidMat
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseNoSplit+sseSplit

        #更新分类和中心点
        ##分类的0,1 类别标记为 第len(centList)类和 一个原类(即是在划分前的某一类)
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)#被二分的结果永远是0,1两类。手动上标记。该类标记为新类别
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit#该类别是原类别

        centList[bestCentToSplit]=bestNewCents[0,:]#原类别的中心变了，因为上一行人为规定了第一行表示原来的集合，第二行是新类的集合
        centList.append(bestNewCents[1,:])#添加新类别的中心
        clusterAssement[nonzero(clusterAssement[:,0].A==bestCentToSplit)[0],:]=bestClustAss#计算距离矩阵的更新
    return centList,clusterAssement


#改变的二分聚类, 问题：如果分类误差变大了则可能在K类之前就停止了
def biKmeans1(dataSet,k,distMeas=distEclud):
    m=shape(dataSet)[0]
    #create initial centroid
    clusterAssement=mat(zeros((m,2)))#第一列类别，第二列距离
    centroid0=mean(dataSet,0).tolist()[0]#第一次的中心是所有点的均值
    centList={0:centroid0}#均值点 样本点的顺序就是他们的类别
    for j in range(m): #每个样本到原始样本的距离
        clusterAssement[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2#第0类并且到点的距离
    while len(centList)<k:#当聚类的类别个数不到K时
        lowestSSE=inf
        for i in range(len(centList)):#每个类别都划分一遍，筛选出最好结果
            currClusterDataSet=dataSet[nonzero(clusterAssement[:,0].A==i)[0],:]#属于第i类的数据集合
            centroidMat,splitClust=kMeans(currClusterDataSet,2,distMeas) #kmean为两类
            sseSplit=sum(splitClust[:,1])#分类之后，各点到各自的聚类中心距离和
            sseNoSplit=sum(clusterAssement[nonzero(clusterAssement[:,0].A!=i)[0],1])#剩余类别，即当前假设第i类划分，则剩余其他类为划分
            # #sseSplit+sseNoSpli 表示的是一部分划分，加上其他未划分的。可以简化为划分前和后的误差比较
            ssePostSplit = sum(splitClust[:, 1])  # 分类之后，各点到各自的聚类中心距离和
            ssePreSplit=sum(clusterAssement[nonzero(clusterAssement[:,0].A==i)[0]])
            #----------
            if ssePostSplit<ssePreSplit:#寻找最优划分，
                bestCentToSplit=i#原类别
                bestNewCents=centroidMat
                bestClust=splitClust.copy()
        #更新分类和中心点
        ##分类的0,1 类别标记为 第len(centList)类和 一个原类(即是在划分前的某一类)
        bestClust[nonzero(bestClust[:,0].A==1)[0],0]=len(centList)#被二分的结果永远是0,1两类。手动上标记。该类标记为新类别
        bestClust[nonzero(bestClust[:,0].A==0)[0],0]=bestCentToSplit#该类别是原类别

        centList[bestCentToSplit]=bestNewCents[0,:]#原类别的中心变了，因为上一行人为规定了第一行表示原来的集合，第二行是新类的集合
        centList[len(centList)]=bestNewCents[1,:]

        clusterAssement[nonzero(clusterAssement[:,0].A==bestCentToSplit)[0],:]=bestClust#计算距离矩阵的更新 bestCentToSplit 是原被划分的类别，一步更新为新类别+距离
    return centList,clusterAssement


if __name__=='__main__':
    datMat=mat(loadDataSet('kmeanTestSet.txt'))
    # print randCent(datMat,5)
    # print distEclud(datMat[:,1],datMat[:,0])
    centroids,clusterClass=biKmeans1(datMat,4)
    print centroids


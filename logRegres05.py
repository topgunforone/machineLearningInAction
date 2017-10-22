#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr =open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    alpha=0.01
    maxCycles=500
    weights=np.ones((n,1))
    one=mat(ones(m)).T
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        error=np.multiply(error,h)
        error=np.multiply(error,one-h)
        weights=weights+alpha*dataMatrix.T*error
    return weights

#随机梯度下降
def stocgradAscent0(dataMatIn,classLabels):
   m,n=shape(dataMatIn)
   alpha=0.1
   weights=ones(n)
   for i in range(m):
       h=sigmoid(sum(dataMatIn[i])*weights)
       error=(classLabels[i]-h)*h*(1-h)
       weights=weights+alpha*error*dataMatIn[i]
       return  weights

#加入不确定性的随机梯度下降
def stocGradAscent1(dataMatIn,classLabels,numIter=150):
   m,n=shape(dataMatIn)
   weights=ones(n)
   for i in range(numIter):
       dataIndex=range(m)
       for j in range(m):
           alpha = 4/(1+i+j)+0.01
           randIndex=int(random.uniform(0,len(dataIndex)))
           h=sigmoid(sum(dataMatIn[randIndex]*weights))
           error=(classLabels[i]-h)*h*(1-h)
           weights=weights+alpha*error*dataMatIn[randIndex]
           del (dataIndex[randIndex])
   return  weights

def plotBestFit(wei):
    weights=wei
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    # xcord1=[];ycord1=[]
    # xcord2=[];ycord2=[]
    # for i in range(n):
     #         xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
    #     else:
    #         xcord2.append(dataArr[i, 1]);ycord2.append(dataArr[i, 2])
    #
    plt.figure()
    plt.scatter(dataArr[:,1],dataArr[:,2],c=labelMat)
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    y=y.T
    plt.plot(x,y)
    plt.xlabel('$x_1$')
    plt.ylabel('$y_1$')
    plt.show()


def classifyVector(inX,weights):
    '''
    inX：行向量
    weights：列向量
    :param inX:
    :param weights:
    :return:
    '''
    prob=sigmoid(sum(inX*weights))
    if prob >0.5:
        return 1.0
    else:return 0.0

def colicTest():
    fr=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in fr.readlines():
        currentLine=line.strip().split('\t')
        trainingSet.append(currentLine[0:-1])
        trainingLabels.append(currentLine[-1])
    trainWeights=stocGradAscent1(trainingSet,trainingLabels)
    return  trainWeights
    # errorCount=0.0;numTesVec=0.0
    # frTest=open('horseColicTest.txt')
    # for line in frTest.readlines():
    #     numTesVec+=1.0
    #     currentLine=line.strip().split('\t')
    #     lineArr=[]
    #     for i in range(21):
    #         lineArr.append(float(currentLine[i]))
    #     if int(classifyVector(array(lineArr),trainWeights))!= int(currentLine[21]):
    #         errorCount+=1.0
    #     errorRate=float(errorCount)/numTesVec
    #     print "the error rate of this test is:%f"% errorRate
    # return errorRate
def multiTest():
    numTest=10.0;errSum=0.0
    for k in range(numTest):
        errSum+=colicTest()
    print "after %d iterations the average error rate is:%f"%(numTest,errSum/numTest)




if __name__=='__main__':
    dataArr,labelMat=loadDataSet()
    # weights=stocGradAscent1(dataArr,labelMat)
    # plotBestFit(weights)
    colicTest()
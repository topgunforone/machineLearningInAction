#-*-coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt


#加载数据
def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr = open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return  dataMat,labelMat


#标准回归
def standRegress(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx) ==0.0:
        print "This matrix is singular,cannot do inverse"
        return
    else:
        ws=xTx.I*(xMat.T*yMat)
    return  ws

#局部加权线性回归函数
def  lwlr(testPoint,xArr,yArr,k=0.1):
    xMat=mat(xArr);yMat=mat(yArr).T#一维取mat会偷懒躺下，需要站着
    m=shape(xMat)[0]
    weights=mat(eye(m))#testPoint 到每个样本点的距离
    for j in range(m):
        diffMat=testPoint-xMat[j,: ]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print "This matrix is singular,cannot do inverse"
        return
    else:
        ws=xTx.I*(xMat.T*(weights*yMat))
    return  testPoint*ws


def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat


#自写, 矩阵类的用np.mat
def lwlrTest01(testArr,xArr,yArr,key=1.0):
    testArr=mat(testArr)
    xArr=mat(xArr)
    yArr=mat(yArr).T
    m=shape(xArr)[0]
    n=shape(testArr)[0]
    yHat=[]
    for i in range(n):
        wMat=(eye(m))
        for j in range(m):
            wMat[j,j]=exp(-sum((testArr[i]-xArr[j]))**2/(2*key))
        theta_Hat=(xArr.T*wMat*xArr).I*xArr.T*wMat*yArr
        yHat.append(testArr*theta_Hat)
    return yHat



def _plot(xArr,yArr,weights):
    plt.figure()
    plt.scatter(mat(xArr)[:, 1], mat(yArr))
    xCopy = mat(xArr).copy()
    xCopy.sort(0)  # sort 按照列排序  如果为1 则是按照行排列
    yHat = xCopy * weights
    plt.plot(xCopy[:, 1], yHat)
    plt.show()

def rssError(yArr,yHatArr):
    '''
    计算误差
    :param yArr: n*1
    :param yHatArr: n*1
    :return:  一个实数
    '''
    return sum((yArr-yHatArr)**2)


def _plot_W(xArr,yArr,yHat1):
    plt.figure()
    xMat=mat(xArr)
    indSort=xMat[:,1].argsort(0)
    yMat=mat(yArr).T
    xSort=xMat[indSort][:,0,:]#将一个[[]]降维[]
    plt.scatter(xSort[:,1],yMat[indSort],s=2)#yMat之后又嵌入一对[]
    plt.plot(xSort[:,1],yHat1[indSort],c='red')
    plt.show()


##################岭回归######################
def ridgeRegres(xMat,yMat,lam=0.2):
    '''
    :param xMat:
    :param yMat:
    :param lam:
    :return:  ws is a  n*1 matrix
    '''
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print "This matrix is singular,cannot do inverse"
        return
    else:
        ws=denom.I*xMat.T*yMat
    return  ws


def  ridgeTest(xArr,yArr):  ``                  `````   `
    #数据的标准化 返回30*属性列个值
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xMean=mean(xMat,0)
    yMean=mean(yMat)
    xVar=var(xMat,0)
    xMat=(xMat-xMean)/xVar
    yMat=(yMat-yMean)/var(yMat,0)
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))#记录每个lambda对应的系数
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-1))
        wMat[i,:]=ws.T
    return  wMat


#############前向逐步线性回归##################
def stepWise(xArr,yArr,eps=0.1,numIt=100):
    xMat=mat(xArr);yMat=mat(yArr).T
    yMean=mean(yMat,0)
    xMean=mean(xMat,0)
    yMat=yMat-yMean
    xMat=(xMat-xMean)/var(xMat,0)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1));wsTest=ws.copy();wsMax=ws.copy()
    for i in range(numIt):
        lowestError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=sign*eps
                err=rssError(yMat,xMat*wsTest)
                if err<lowestError:
                    lowestError=err
                    wsMax=wsTest
        returnMat[i,:]=wsMax.T
    return  returnMat



def stageWise(xArr,yArr,eps=0.1,numIt=100):
#标准化数据
    xMat=mat(xArr);yMat=mat(yArr).T
    yMean=mean(yMat,0)
    xMean=mean(xMat,0)
    yMat=yMat-yMean
    xMat=(xMat-xMean)/var(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1));wsTest=ws.copy();wsMax=ws.copy()
#每一轮迭代中
    for i in range(numIt):
        # print  ws.T
        lowestError=inf#设置最小误差
        for j in range(n):#对每一个特征
            for sign in [-1,1]: #增大或缩小一个系数
                wsTest=ws.copy()
                # 每轮通过比较每个特征增加或者减少两次的误差大小，取较小者为改变的最后系数
                wsTest[j]+=eps*sign#步长
                yTest=xMat*wsTest#计算新W下误差
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return  returnMat



def crossValidation(xArr,yArr,numVal=10):
    m=len(yArr)
    indexList=range(m)#used to random
    errorMat=zeros((numVal,30))#30 is the times superparameters
    for i  in range(numVal):#10折
        trainX=[];trainY=[]
        testX=[];testY=[]
        random.shuffle(indexList)
        for j in range(m):
            if j <m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat=ridgeTest(trainX,trainY)#30*weightsNum
        for k in range(30):
            matTestX=mat(testX);matTrainX=mat(trainX)
            meanTrain=mean(matTestX,0)
            varTrain=var(matTrainX,0)
            matTestX=(matTestX-meanTrain)/varTrain
            yEst=matTestX*mat(wMat[k,:]).T+mean(trainY)
            errroMat[i,k]=rssError(yEst.T.A,testY.A)#10*error
    meanErrors=mean(errorMat,0)
    minMean=float(min(meanErrors))
    bestWeights=wMat[nonzero(meanErrors==minMean)]#不适用，换用.index
    xMat=mat(xArr);yMat=mat(yArr).T
    meanX=mean(xMat,0);varX=var(xMat,0)






if __name__=='__main__':
    xArr,yArr=loadDataSet('ex0.txt')
    weights=standRegress(xArr,yArr)
    print weights
    #     yHat=mat(xArr)*weights
#     # _plot(xArr,yArr,weights)
#     # print corrcoef(yHat.T,mat(yArr))
#     #注意系数大小和拟合的结果关系
#     kernalRate=3
#     yHat1 =lwlrTest(xArr,xArr,yArr,kernalRate)#得到预测的结果，权重是根据每一个具体的数而得到的。
#     _plot_W(xArr, yArr, yHat1)
# # #鲍鱼的实例
    abX,abY=loadDataSet('abalone.txt')
    # yHat2=lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
#     print rssError(yHat2.T,abY[100:199])

# #
#     ridgeWeights=ridgeTest(abX,abY)
#     plt.figure()
#     plt.plot(ridgeWeights)
#     plt.show()
#     print stepWise(abX,abY,0.001,5000)
    print lwlrTest01(abX[100:199],abX[0:99],abY[0:99],0.1)

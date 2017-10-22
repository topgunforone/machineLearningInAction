#coding:utf-8
from numpy import *
def loadSimData():
    datMat=matrix([[1.0,2.1],[2.0,1.1],[1.3,1.0],
                   [1.0,1.0],[2.0,1.0]])
    classLabels=[1.0,1.0,1.0,-1.0,1.0]
    return  datMat,classLabels

#决策树桩
def stumpClassify(dataMatrix,dimen,thereshVal,thereshIneq):
    '''
    决策树桩
    :param dataMatrix:
    :param dimen:
    :param thereshVal:
    :param thereshIneq:
    :return:  返回 分类的n*1向量 值取[-1,1]
    '''
    retArray=ones((shape(dataMatrix)[0],1))
    if thereshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=thereshVal]=-1.0
    else:
        retArray[dataMatrix[:, dimen] >thereshVal] = -1.0
    return  retArray


#构建一个决策树弱分类器
def buildStump(dataArr,classLabels,D):
    '''
    :param dataArr:
    :param classLabels:
    :param D:数据的权重向量
    :return:
    '''
    dataMatrix = mat(dataArr);
    m, n = shape(dataMatrix)
    labelMat=mat(classLabels).T
    minEror=inf
    numSteps=10
    bestStump={}
    bestClassEst=mat(zeros((m,1)))
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1): #从该连续属性的每个分化点开始遍历
            for inequal in ['lt','gt']:
                threshVal=rangeMin+stepSize*float(j)
                predictVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))#如果判断错误，错误结果1*weight
                errArr[predictVals==labelMat]=0
                weightedErr=D.T*errArr#计算加权错误率
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %f" % (
                # i, threshVal, inequal, weightedErr)
                if weightedErr<=minEror:
                    minEror=weightedErr
                    bestClassEst=predictVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minEror,bestClassEst

#基于单层决策树的AdaBoosting 训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        # print 'D: ', D.T
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        # print 'classEst  ' ,classEst.T
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        # print 'aggClassEst:',aggClassEst.T
        aggErros=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErros.sum()/m
        # print 'total error:',errorRate
        if errorRate==0:
            break
    return weakClassArr


#adaBoost的分类函数
def adaClassify(datToClass,classifierArr):
    dataMatrix=mat(datToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                  classifierArr[i]['thresh'],\
                                  classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print '每次决策后的分类结果  ', aggClassEst
    return  sign(aggClassEst)





if __name__=='__main__':
    datMat,classLabels=loadSimData()
    # buildStump(datMat,classLabels,mat(ones((5,1))/5))
    classifierArr=adaBoostTrainDS(datMat,classLabels,40)
    print adaClassify([0,0],classifierArr)
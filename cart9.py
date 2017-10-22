#coding:utf-8
from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def regLeaf(dataSet):#不再对数据进行切分时，调用该函数得到叶节点的模型
    return(mean(dataSet[:,-1]))#回归树中，目标变量的均值

def regErr(dataSet):
    return(var(dataSet[:,-1])*(shape(dataSet)[0]))#返回总方差

def binSplitDataSet(dataSet,feature,value):
    '''
    根据某一属性， 二分样本集合
    :param dataSet:
    :param feature:
    :param value:
    :return: < 在左边，大于在右边
    '''
    mat0=dataSet[nonzero((dataSet[:,feature])>value)[0],:]#此处下上<>的变化，对结果有轻微影响
    mat1 = dataSet[nonzero((dataSet[:, feature]) <= value)[0], :]
    return mat0,mat1

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0]#容许的误差下降值
    tolN=ops[1]#切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:#如果类别用set去重后，只有一个类别
        return  None,leafType(dataSet)
    m,n=shape(dataSet)
    S=errType(dataSet)#切分前误差
    bestS=Inf;bestIndex=-1;bestValue=-1
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]) :
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):continue#少于最小切分数，停止切分
            newS=errType(mat0)+errType(mat1)#切分后误差
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if abs(S-bestS)<tolS:#遍历完后发现下降少于1  则停止划分
        return  None,leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):#如果切分后满足停止条件，返回结果
        return  None,leafType(dataSet)
    return bestIndex,bestValue  #返回切分特征和特征值




def creatTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''
    建立数
    :param dataSet:
    :param leafType: 建立叶节点的函数
    :param errType:   误差计算函数
    :param ops:  树构建所需其他参数的元组
    :return:
    '''
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:return  val#如果是回归树，该模型是一个常数，如果是模型数，其模型为一个线性方程
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)#划分成两部分数据集
    retTree['right'] = creatTree(rSet, leafType, errType, ops)
    retTree['left']=creatTree(lSet,leafType,errType,ops)
    return  retTree

#判断当前处理的节点是否是叶节点
def isTree(object):
    return type(object).__name__ =='dict'

#如果找到两个叶节点，则计算他们的平均值
# def getMean(tree):
#     if isTree(tree['right']):tree['right']=getMean(tree['right'])
#     if isTree(tree['left']): tree['left'] = getMean(tree['left'])
#     return (tree['left']+tree['right'])/2.0

def getMean(tree):
    if (not isTree(tree['left'])) and (not isTree(tree['right'])):
        return (tree['left'] + tree['right']) / 2.0
    if  isTree(tree['right']):
         tree['right']=getMean(tree['right'])
         return tree['right']
    if  isTree(tree['left']):
        tree['left']=getMean(tree['left'])
        return tree['left']

def prune(tree,testData):
    '''
    剪枝
    :param tree:待剪枝的树，通过训练集训练出来的模型
    :param testData: 测试数据集
    :return: 返回剪枝的结果
    '''
    if shape(testData)[0]==0:return  getMean(tree)#因为是测试集，可能没有相应的属性值，这种情况进行坍塌处理。即将训练模型的结果兼剪枝到此节点作为叶子节点
    if (isTree(tree['right'])) or (isTree(tree['left'])):#如果接下来的是子树，就按照当前的划分规则划分下去
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']): #如果两边同时满足已经不是子树，开始进行合并
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=sum(power(lSet[:,-1]-tree['left'],2))+\
        sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean=(tree['left']+tree['right'])/2.0 #简化了，不符合原理
        errorMerge=sum(power(testData[:,-1]-treeMean,2))
        if errorNoMerge>errorMerge:#如果合并后的误差小，则进行合并，否则直接返回，不合并
            print 'merging'
            return treeMean
        else:return tree
    else:return  tree


def linearSolve(dataSet):
    m,n =shape(dataSet)
    X=mat(ones((m,n)));Y=mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1];Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise  NameError('This matrix is singulsr,connt do inverse')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return  ws

def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))


#用树回归进行预测
def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:(n+1)]=inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):return  modelEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return  treeForeCast(tree['left'],inData,modelEval)
        else:
            return  modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return  treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)



def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat

if __name__=='__main__':
    # testMat=mat(eye(4))
    # print binSplitDataSet(testMat,1,0.5)
    # myDat=loadDataSet('ex0.txt')
    # myDat=mat(myDat)
    # # print creatTree(myDat)
    # # plt.figure()
    # # plt.scatter(myDat[:,0],myDat[:,1])
    # # plt.scatter([0.48813]*ones((10,1)),linspace(-1,1,10))
    # # plt.show()
    # myMat2=mat(loadDataSet('ex2test.txt'))
    # myTree=creatTree(myDat,ops=(0,1))
    # # dataSet=loadDataSet('ex2test.txt')
    # dataSet=mat(dataSet)
    # trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    # testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    # mytree=creatTree(trainMat,ops=(1,20))
    # yHat=createForeCast(mytree,testMat[:,0])
    # corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    # # print yHat
    # # print mytree
    # mytree=creatTree(trainMat,modelLeaf,modelErr,(1,20))
    # yHat1=createForeCast(mytree,testMat[:,0],modelTreeEval)
    #
    # print mytree
    a = {'spInd': 1, 'spVal': matrix([[0.39435]]), 'right': {'spInd': 1, 'spVal':
        matrix([[0.197834]]), 'right': -0.023838155555555553, 'left':1.0289583666666664},
         'left': {'spInd': 1, 'spVal': matrix([[0.582002]]),
                  'right': 1.9800350714285717, 'left': {'spInd': 1, 'spVal': matrix([[
                 0.797583]]), 'right': 2.9836209534883724, 'left': 3.9871632000000004}}}
    print getMean(a)
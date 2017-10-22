#-*-coding:utf-8 -*-
from math import log
import pickle
def create_DataSet():
    dataSet=[[1,1,'yes'],
          [1,1,'yes'],
          [1,0,'no'],
          [0,1,'no'],
          [0,1,'no']]
    labels=['no surfacing','flipper']
    return  dataSet,labels
def calcShannoEnt(dataSet):
    numEntries=len(dataSet)
    labelCount={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        labelCount[currentLabel]=labelCount.get(currentLabel,0)+1
    shannonEnt=0.0
    for key in labelCount:
        prob=float(labelCount[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    '''
    dataSet 的第 axis列，如果取值等于value则挑选出该行，并且删除第axis列的那个值
    :param dataSet:
    :param axis:
    :param value:
    :return:
    '''
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeaVec=featVec[:axis]
            reducedFeaVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeaVec)
    return retDataSet

#选择最好的数据集合划分,返回分划最好的属性列
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannoEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannoEnt(subDataSet)
        InfoGain=baseEntropy-newEntropy
        if (InfoGain>bestInfoGain):
            bestInfoGain=InfoGain
            bestFeature=i
    return  bestFeature

def majorityCnt(classList):
    '''
    返回类别次数最多的类别名
    :param classList:
    :return:
    '''
    classCount={}
    for vote in classList:
        classCount[vote]=classCount.get(vote,0)+1
    sortedClassCount=sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #如果种类都一样
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:#如果属性划分完毕，返回最多的类别
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)#返回的是第几个特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])# 在此处删除了一个属性
    featValues=[example[bestFeat]for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),\
        subLabels)
    return myTree


#决策树的存储
def storeTree(inputTree,filename):
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    fr =open(filename)
    return pickle.load(fr)
#############决策树做隐形眼镜###########3333
def openfile(filename):
    lensens=[]
    with open(filename) as f:
        fr=f.readlines()
        for i in fr:
            lensens.append(i.strip().split('\t'))
    return lensens

if __name__=='__main__':
    myDat,labels=create_DataSet()
    # print myDat
    # print calcShannoEnt(myDat)
    # print chooseBestFeatureToSplit(myDat)
    # mytree = createTree(myDat,labels)
    # storeTree(mytree,'mytree')
    # print pickle.load(open('mytree'))
    lensens=openfile('lenses.txt')
    # print lensens
    lensensLabels=['age','prescript','astigmatic','tearRate']
    lensesTree=createTree(lensens,lensensLabels)
    print lensesTree
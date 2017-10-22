#-*-coding:utf-8 -*-
import numpy as np
import random
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    '''
    不重复的字典
    :param dataSet:
    :return:
    '''
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList,inputSet):
    '''
    转为词向量
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else: print "the word:%s is no in vocabulary" %word
    return returnVec


def bagOfWords2VecMN(vocabList,inputSet):
    '''
    转为词向量
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else: print "the word:%s is no in vocabulary" %word
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    '''
    训练文档，以向量的形式输出 log p(x_i|c=1),log p(x_i|c=0)和p(c=1)的概率
    :rtype: object
    :param trainMatrix:
    :param trainCategory:
    :return:
    '''
    numTrainDocs=len(trainMatrix)#文章数
    numWords=len(trainMatrix[0])#词典长度
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    # p0Num=zeros(numWords)
    # p1Num=zeros(numWords)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)#分子舵加一个
    p0Denom=2.0
    p1Denom=2.0#平滑操作 分母由于是二分类，取两个
    #计算p（w_i|c）
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect=p1Num/p1Denom
    # p0Vect=p0Num/p0Denom
    p1Vect = np.log(p1Num / p1Denom)#（w_i|c）
    p0Vect = np.log(p0Num / p0Denom)#（w_i|c）
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry=['1love','1my','1dalmation']#如果没有的词为0 但是有平滑保证
    #如果输入的数字在词典里没有，那么更具分类结果给中谁的类别多来判断，即完全的先验概率
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)



###############过滤垃圾邮件##############################
def textParse(bigString):
    # import re
    listOfTokens=bigString.split()
    return[tok.lower() for tok in listOfTokens  if len(tok)>2]
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    #打开文件
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)#一个list
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)  # 一个list
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
#50个训练
    trainingSet=range(50)
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0.0
    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) !=classList[docIndex]:
            errorCount+=1
            print' classifie erro is ', docList[docIndex]
    print 'the error rate is ' ,float(errorCount)/len(testSet)








if __name__=='__main__':
    # testingNB()
    spamTest()
#-*-coding:utf-8 -*-
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
def CreateDataSet():
    group=array([[1.0,1.1][1.0,1.0][0,0][0,0.1]])
    labels=['A','A','B','B']
    return group,labels


def classify0(inX,dataSet,labels,k):
    '''
    classify
    :param inX:to be classified vector
    :param dataSet: training set
    :param labels: labels
    :param k:  nearese number
    :return:
    '''
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndices=distances.argsort()# rank of the  value
    #reverse using: np.argsort(-a)
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    # sortedClassCount=sorted(classCount.iteritems(),\
                                # key=operator.itemgetter(1),reverse=True)
    sortedClassCount=sorted(classCount.items(),\
                                key=lambda x:x[1],reverse=True)
    return  sortedClassCount[0][0]


 #reading data
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in  arrayOLines:
        line=line.strip().split('\t')
        returnMat[index,:]=line[0:3]
        classLabelVector.append(line[-1])
        index+=1
    return returnMat,classLabelVector

#dataSet are continous variables
def  autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return  normDataSet,ranges,minVals

#hoRatio% as test, rest as train
def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                datingLabels[numTestVecs:m],3)
        print "the classifier came back with:%s,the real answer is:%s" % (classifierResult,datingLabels[i])
        if (classifierResult!=datingLabels[i]):errorCount+=1
    print "the total error rate is %f" %(errorCount/float(numTestVecs))

    #using an examle
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percenTats=float(raw_input("percentage of time spent playing games?"))
    ffMiles=float(raw_input("frequent filer miles earned per year?"))
    icecream=float(raw_input("liters of ice cream consumed per year?"))
    inArr=array([ffMiles,percenTats,icecream])
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print 'you will probably like this person:',resultList[int(classifierResult)-1]


####################Hand_Writing#############################
def img2vector(filename):
    '''
    translate 32*32 to 1*1024
    :param filename:
    :return:
    '''
    returnVect=zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('digits/trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNameStr=int(fileStr.split('_')[0])
        hwLabels.append(classNameStr)
        trainingMat[i,:]=img2vector('digits/trainingDigits/%s'\
                                    %fileNameStr)
    testFileList=listdir('digits/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('digits/testDigits/%s'%fileNameStr)
        classfierResult=classify0(vectorUnderTest,\
                                  trainingMat,hwLabels,3)
        print "the classifier came back with:%d,the real answer is:%d"\
        %(classfierResult,classNumStr)
        if classfierResult != classNumStr :errorCount+=1.0
    print "\nthe total number of erros is:%d" %errorCount
    print "\nthe total error rate is:%f" %(errorCount/float(mTest))

if __name__=='__main__':
    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    # knn=kNN()
    # group,labels=knn.CreateDataSet()
    # print group,labels
    # print classify0([0,0],group,labels,3)
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    #plot
    plt.figure(facecolor='w')
    plt.scatter(datingDataMat[:,1],datingDataMat[:,2],c=datingLabels)
    plt.xlabel('play')
    plt.ylabel('icecream')
    plt.legend(bbox_to_anchor=(1.02,0.8),loc='center left')
    plt.show()
    # datingClassTest()
    # classifyPerson()
    # path=u"E:\课件\code\machineLearningInAction\digits\\trainingDigits\\0_0.txt"
    # print img2vector(path)
    # handwritingClassTest()
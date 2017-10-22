#coding:utf-8
from numpy import*
def loadDatSet():
    return mat([[0,0,0,2,2],[0,0,0,1,1],[1,1,1,1,0],
                [2,2,2,1,0],[5,5,5,1,0],[1,1,1,1,1]])



#相似度的计算
def ecludSim(inA,inB):
    return 1.0/(1.0+linalg.norm(inA-inB))

def pearsSim(inA,inB):#列向量
    if len(inA)<3:
        return 1
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]
def cosSim(inA,inB):
    num=float(inA.T*inB)
    nom=linalg.norm(inA)*linalg.norm(inB)
    return  0.5+0.5*(num/nom)

#基于物品相似度的推荐引擎
def standEst(dataMat,user,simMeas,item):
    '''
    :param dataMat:数据矩阵 用户*物品矩阵
    :param user: 用户标号
    :param simMeas: 相似度计算方法方法
    :param item: 物品编号
    :return:返回的是每个物品的评分*每个物品相似度=目前item的评分
    '''
    n=shape(dataMat)[1]#用户数
    simTotal=0.0;ratSimTotal=0.0
    for  j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:continue
        overLap=nonzero(logical_and(dataMat[:,item].A1>0,dataMat[:,j].A1>0))[0]
        if len(overLap)==0:similarity=0
        else:
            similarity=simMeas(dataMat[overLap,item],dataMat[overLap ,j])#切出值只有非零的公共部分
            simTotal+=similarity
            ratSimTotal+=similarity*userRating
    if simTotal==0:return 0
    else:return  ratSimTotal/simTotal



def command(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:return 'you rated everthing'
    itemSocre=[]
    for item in unratedItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemSocre.append((item,estimatedScore))
    return sorted(itemSocre,key= lambda jj:jj[1],reverse=True)[:N]

def svdEst(datMat,user,simMeas,item):
    n=shape(datMat)[1]
    simTotal=0.0;ratSimTotal=0.0
    U,sigma,VT=linalg.svd(datMat)
    Sig4=mat(eye(4)*sigma[:,4])
    XformedItems=dataMat.T*U[:,:4]*Sig4.I
    for  j in range(n):
        userRating=datMat[user,j]
        if userRating==0 or j==item: continue
        similarity=simMeas(XformedItems[item,:].T,XformedItems[j,:].T)
        simTotal+=similarity
        ratSimTotal+=userRating*similarity
    if similarity==0:return 0
    else: return  ratSimTotal/simTotal

if __name__=='__main__':
    datMat=loadDatSet()
    # #svd分解及重构
    # U,sigma,v=linalg.svd(datMat)
    # print 'u is','\n' ,U,'\n'
    # print 'sigma is','\n',sigma,'\n'
    # print 'v is','\n', v, '\n'
    # print u'利用前两个奇异值重构矩阵','\n'
    # sigma2=[[sigma[0],0],[0,sigma[1]]]
    # reconsMat=datMat.T*U[:,:2]*mat(sigma2)
    # print reconsMat


    #物品协同过滤对未评分返回前N项
    command(datMat,2)
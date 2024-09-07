'''
'''
import math
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.ndimage import filters
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import numpy.linalg as lg
### FUNCTIONS #############################################################################################
from typing import Any
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout
from scipy.io import savemat, loadmat

def GenerateMeshAndDisMat(Field_x_start,Field_x_end,NSx,Field_z_start,Field_z_end,NSz,nodeNumPerX,nodeNumPerZ):
    t1 = time.time()
    elementNodeLocationZ = np.linspace(Field_z_start, Field_z_end, NSz + 1)
    elementNodeLocationX = np.linspace(Field_x_start, Field_x_end, NSx + 1)
    elementDeltaX = np.diff(elementNodeLocationX)
    elementDeltaZ = np.diff(elementNodeLocationZ)
    [meshNodeNumPerX, meshNodeNumPerZ] = np.meshgrid(nodeNumPerX, nodeNumPerZ)
    meshNodeNumPerXVec = np.reshape(meshNodeNumPerX,(-1,1))
    meshNodeNumPerZVec = np.reshape(meshNodeNumPerZ,(-1,1))
    nodeNumPerZ = nodeNumPerZ.astype(int)
    nodeNumPerX = nodeNumPerX.astype(int)
    intervalNumPerZ = nodeNumPerZ + 1
    intervalNumPerX = nodeNumPerX + 1
    nodeDeltaX = elementDeltaX / intervalNumPerX
    nodeDeltaZ = elementDeltaZ / intervalNumPerZ
    nodeLocationZTemp1 = -1 * np.ones((20, NSz))
    nodeLocationXTemp1 = -1 * np.ones((20, NSx))
    # nodeLocationZTemp1 = np.zeros((intervalNumPerZ[0],NSz))
    for i in range(NSz):
        for j in range(intervalNumPerZ[i]):
            # aa= elementNodeLocationZ[i] + (j+1) * nodeDeltaZ[i]
            nodeLocationZTemp1[j,i] = elementNodeLocationZ[i] + (j+1) * nodeDeltaZ[i]
    nodeLocationZTemp2 = np.reshape(nodeLocationZTemp1,(-1,1), order='f')
    nodeLocationZ = np.concatenate(([elementNodeLocationZ[0]],nodeLocationZTemp2[nodeLocationZTemp2 != -1]))
    for i in range(NSx):
        for j in range(intervalNumPerX[i]):
            nodeLocationXTemp1[j, i] = elementNodeLocationX[i] + (j+1) * nodeDeltaX[i]
    nodeLocationXTemp2 = np.reshape(nodeLocationXTemp1,(-1,1),order='f')
    nodeLocationX = np.concatenate(([elementNodeLocationX[0]],nodeLocationXTemp2[nodeLocationXTemp2 != -1]))
    Nx = len(nodeLocationX)
    Nz = len(nodeLocationZ)
    allLabel = np.zeros((1, Nz*Nx))
    allLabel[0] = np.linspace(0,Nz*Nx-1,Nz*Nx)
    allLabel = np.reshape(allLabel, (Nz, Nx),order='f')
    [XMat, ZMat] = np.meshgrid(nodeLocationX, nodeLocationZ)
    XVec = np.reshape(XMat,(-1,1),order='f')
    ZVec = np.reshape(ZMat,(-1,1),order='f')
    activeNodeNumMat = np.zeros((50, NSz * NSx)) # why50?  保证每个大网格的四周声学点个数小于50% 列数等于大网格数
    labelZ = np.cumsum(np.concatenate(([0], intervalNumPerZ)))
    labelX = np.cumsum(np.concatenate(([0], intervalNumPerX)))
    labelX = np.cumsum(np.concatenate(([0], intervalNumPerX)))
    index = 0
    for i in range(NSx):
        for j in range(NSz):
            currentlabelZ = labelZ[j]
            currentlabelX = labelX[i]
            # 只算网格上下边界上的格点
            aa=[currentlabelZ, currentlabelZ + intervalNumPerZ[j]]
            bb=(np.linspace(currentlabelX + 1,currentlabelX + nodeNumPerX[i],nodeNumPerX[i]).astype(int)).tolist()
            temp = allLabel[aa*3, [val for val in bb for i in range(2)]]
            # temp = allLabel[[currentlabelZ, currentlabelZ + intervalNumPerZ[j]], range(currentlabelX + 1,currentlabelX + nodeNumPerX[i] + 1)]
            temp1 = np.reshape(temp,(-1,1),order='f')
    # 只算网格左右边界上的格点
            aa = [currentlabelX, currentlabelX + intervalNumPerX[i]]
            temp = allLabel[2*(np.linspace(currentlabelZ + 1,currentlabelZ + nodeNumPerZ[j],nodeNumPerZ[j]).astype(int)).tolist(),
                            [val for val in aa for i in range(3)]]
            temp2 = np.reshape(temp,(-1,1),order='f')
            temp = np.concatenate((temp1,temp2))
            activeNodeNumMat[0 : len(temp) , index] = np.reshape(temp,-1)
            index = index + 1
    sortActiveNodeNum = np.reshape(activeNodeNumMat,(-1,1), order='f')
    sortActiveNodeNum = np.unique(sortActiveNodeNum)
    infoTable = np.zeros((len(sortActiveNodeNum) -1, 6))
    infoTable[:, 0] = sortActiveNodeNum[1: ] # infoTable(:, 1) : 活跃的声学点序号
    infoTable[:, 1] = np.arange(0,np.size(infoTable,0))  #
    infoTable[:, 2] = np.reshape(XVec[np.reshape(infoTable[:, 0],-1).astype(int)],-1) # infoTable(:, 3):找到活跃声学点横坐标
    infoTable[:, 3] = np.reshape(ZVec[np.reshape(infoTable[:, 0],-1).astype(int)],-1) # infoTable(:, 4):找到活跃声学点纵坐标
    sparseIndex = 0
    # timestamp1 = time.time()
    distance = np.array([])
    sparseRow = np.array([])
    sparseCollum = np.array([])
    for i in range(np.size(activeNodeNumMat, 1)):
        currentElementNode = activeNodeNumMat[:, i] # activeNumMat: 每个大网格周围的12个点的序号
        currentElementNode = currentElementNode[currentElementNode != 0]
        for p in range(len(currentElementNode)):
            for q in range(len(currentElementNode)):
                if (abs(currentElementNode[p] - currentElementNode[q]) >= meshNodeNumPerZVec[i]
                 and (math.floor(abs(currentElementNode[p] - currentElementNode[q]) / Nz) == 1 + meshNodeNumPerZVec[i]
             or abs(currentElementNode[p] - currentElementNode[q]) % Nz != 0)):
                # 一个大网格内选择路径的逻辑：连线经过网格的内部
                    globalIndexS = int(infoTable[infoTable[:, 0] == currentElementNode[p], 1])
                    globalIndexR = int(infoTable[infoTable[:, 0] == currentElementNode[q], 1])
                    distance = np.append(distance, math.sqrt((infoTable[globalIndexS, 2] - infoTable[globalIndexR, 2]) ** 2
                                 + (infoTable[globalIndexS, 3] - infoTable[globalIndexR, 3]) ** 2))
                    sparseRow = np.append(sparseRow, globalIndexS)
                    sparseCollum = np.append(sparseCollum, globalIndexR)
                    # weight(sparseIndex) = distance * slowness(i)
                    # sparseIndex = 1 + sparseIndex

    disMat = csr_matrix((distance, (sparseRow, sparseCollum)), shape=(len(sortActiveNodeNum) - 1 , len(sortActiveNodeNum) - 1 )).toarray()

    mapNodeGlobal2Active = np.zeros(infoTable[-1,0])
    for j in range(len(infoTable[:,0])):
       mapNodeGlobal2Active[infoTable[j,0]] = j

    bigGridN = len(currentElementNode)
    infoTable[:,4:5] = 0
    for j in range(len(activeNodeNumMat[0,:])):
        for k in range(bigGridN):
            if(infoTable[mapNodeGlobal2Active[activeNodeNumMat[k,j] - 1] - 1,4]==0):
                infoTable[mapNodeGlobal2Active[activeNodeNumMat[k,j] - 1] - 1,4] = j + 1
            else:
                infoTable[mapNodeGlobal2Active[activeNodeNumMat[k,j] - 1] - 1,5] = j + 1
            end
        end
    end

    t2 = time.time()
    print("GenerateMeshAndDisMat finished, time = {} s".format(t2 - t1))

    return disMat, infoTable, activeNodeNumMat, sparseRow, sparseCollum, mapNodeGlobal2Active

def TxAndRxIndex(Sn,Rn,infoTable):
    surfaceNode = infoTable[infoTable[:, 3] == 0,:]
    # Sn = [20:80: 1000]
    SnNodeIndex = np.zeros(len(Sn))
    for i in range(len(Sn)):
        # i = 2
        surfaceDistance = abs(Sn[i] - surfaceNode[:, 2])
        nearestDistance = min(surfaceDistance)
        nearestNode = surfaceNode[surfaceDistance == nearestDistance, 1]
        if len(nearestNode) > 1:
            nearestNode = nearestNode[0]
        SnNodeIndex[i] = nearestNode

    # Rn = [3000:200: 5000]
    RnNodeIndex = np.zeros( len(Rn))
    for i in range(len(Rn)):
        # i = 2
        surfaceDistance = abs(Rn[i] - surfaceNode[:, 2])
        nearestDistance = min(surfaceDistance)
        nearestNode = surfaceNode[surfaceDistance == nearestDistance, 1]
        if len(nearestNode) > 1:
            nearestNode = nearestNode[0]
        RnNodeIndex[i] = nearestNode
    return SnNodeIndex, RnNodeIndex

def imageExpand(x, expandX, expandY):
    xNumber = np.size(x, 1)
    yNumber = np.size(x, 0)
    xNumberAll = xNumber + 2 * expandX
    yNumberALL = yNumber + 2 * expandY
    y = np.zeros((yNumberALL, xNumberAll))
    y[expandY : yNumber + expandY, expandX : xNumber + expandX] =x   # xNumber + expandX
    y[: expandY, expandX : xNumber + expandX] = np.tile(np.reshape(x[0,:],(1,-1)), (expandY, 1))
    y[yNumber + expandY: , expandX : xNumber + expandX] = np.tile(np.reshape(x[-1,:],(1, -1)), (expandY, 1))
    y[:, : expandX] = np.tile(np.reshape(y[:, expandX],(-1,1)), (1, expandX))
    y[:, xNumber + expandX: ] = np.tile(np.reshape(y[:, xNumberAll - expandX],(-1,1)), (1, expandX))
    return y


def SetUpInvMeshParameter(fieldXStart, domainDistance, XNumberInv, ZNumberInv, initiallength, ratio):
    xEdgeLocationInv = np.linspace(fieldXStart, domainDistance, XNumberInv + 1)
    xElementLocationInv = 0.5 * (xEdgeLocationInv[0:-1] + xEdgeLocationInv[1:])
    hI = np.zeros(ZNumberInv)
    for i in range(ZNumberInv):
        hI[i] = initiallength * ratio ** i
    zEdgeLocationInv = np.concatenate(([0], np.cumsum(hI)))
    zElementLocationInv = 0.5 * (zEdgeLocationInv[:-1] + zEdgeLocationInv[1:])
    [xInv, yInv] = np.meshgrid(xElementLocationInv, -zElementLocationInv)
    [xElementLengthInv, zElementLengthInv] = np.meshgrid(np.diff(xEdgeLocationInv), np.diff(zEdgeLocationInv))
    elementSizeInv = xElementLengthInv * zElementLengthInv
    gridNumberInv = ZNumberInv * XNumberInv
    invMeshPa = {'xInv':xInv, 'yInv':yInv, 'zElementLocationInv':zElementLocationInv, 'xElementLocationInv':xElementLocationInv,
      'gridNumberInv':gridNumberInv,  'ZNumberInv' : ZNumberInv,  'XNumberInv':XNumberInv, 'zEdgeLocationInv':zEdgeLocationInv}
    return invMeshPa

'''InvMeshParameter.xInv = xInv
InvMeshParameter.yInv = yInv
InvMeshParameter.zElementLocationInv = zElementLocationInv
InvMeshParameter.xElementLocationInv = xElementLocationInv
InvMeshParameter.gridNumberInv = gridNumberInv
InvMeshParameter.ZNumberInv = ZNumberInv
InvMeshParameter.XNumberInv = XNumberInv'''

def fspecial_gaussian(p2,p3):
    siz = (p2 - 1) / 2
    std = p3

    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x * x + y * y) / (2 * std * std)
    eps = 2.2204e-16
    h = np.exp(arg)
    h[h < eps * np.max(np.reshape(h, (-1,1),order='f'))] = 0

    sumh = np.sum(np.reshape(h,(-1,1),order='f'))
    if sumh != 0:
        h = h / sumh
    return h


def interp2_nearest(xInv, yInv, fieldRhoInv, xMT, yMT):
    # 后两参数是待插值的坐标  前两参数为插值前的坐标
    ret = np.zeros((np.size(yMT,0),np.size(yMT,1)))
    minpstz = np.zeros(np.size(xMT,0), dtype='int32')
    minpstx = np.zeros(np.size(xMT,1), dtype='int32')
    for i in range(np.size(yMT,0)):
        minpstz[i] = np.argmin(abs(yInv[:, 0] - yMT[i][0]))
    for j in range(np.size(xMT,1)):
        minpstx[j] = np.argmin(abs(xInv[0,:]-xMT[0,j]))
    for i in range(np.size(yMT,0)):
        for j in range(np.size(yMT,1)):
            ret[i,j] = fieldRhoInv[minpstz[i], minpstx[j]]
    return ret



def generatePolygon(x_array, y_array, A, B, C, D):
    [x_vec, y_vec] = np.meshgrid(x_array, y_array)
    # y_vec = -y_vec
    x_vec = np.reshape(x_vec, (-1,1),order='f')
    y_vec = np.reshape(y_vec, (-1,1),order='f')
    x_vec = np.reshape(x_vec,-1)
    y_vec = np.reshape(y_vec,-1)
    # the vector of PA, PB, PC
    PA = np.zeros((len(x_vec), 3))
    PB = PA.copy()
    PC = PA.copy()
    PD = PA.copy()
    PA[:, 0] = A[0] - x_vec
    PA[:, 1] = A[1] - y_vec
    PB[:, 0] = B[0] - x_vec
    PB[:, 1] = B[1] - y_vec
    PC[:, 0] = C[0] - x_vec
    PC[:, 1] = C[1] - y_vec
    PD[:, 0] = D[0] - x_vec
    PD[:, 1] = D[1] - y_vec

    # the cross product of PA, PB, PC
    t1 = np.sign(np.cross(PA, PB)) + 1
    t1 = t1[:, 2]
    t2 = np.sign(np.cross(PB, PC)) + 1
    t2 = t2[:, 2]
    t3 = np.sign(np.cross(PC, PD)) + 1
    t3 = t3[:, 2]
    t4 = np.sign(np.cross(PD, PA)) + 1
    t4 = t4[:, 2]

    # generate a triangle
    mask = np.zeros(len(t1))
    for i in range(len(t1)):
        if(t1[i] !=0 and t2[i] !=0 and t3[i] !=0 and t4[i]!=0):
            mask[i] = 1
        else:
            mask[i] = 0
    return mask

def GenerateLayerSample(xElementLocation, zElementLocation, Dz1,Dz2,backX,baseX, HH = 0, hhLength = 0, hhThickness = 0):
    lgrho_layered = np.reshape(backX,(-1,1),order='f')
    A = np.array([xElementLocation[0]-100,Dz1])
    B = np.array([xElementLocation[-1]+100,Dz2])
    C = np.array([xElementLocation[-1]+100,zElementLocation[-1]+1000])
    D = np.array([xElementLocation[0]-100,zElementLocation[-1]+1000])
    mask = generatePolygon(xElementLocation, zElementLocation, A, B, C, D)
    # add hump and hollow, with mask
    if HH == 1:  # hollow
        vxmin = random.uniform(xElementLocation[14], xElementLocation[-15]-hhLength)
        vxmax = vxmin + hhLength
        v3x = random.uniform(vxmin, vxmax)
        v4x = random.uniform(vxmin, vxmax)
        #################################可以改为当地的层高度
        localdpt = Dz1 + (Dz2 - Dz1)/(B[0] - A[0])*(vxmin - A[0])
        localdpt1 = Dz1 + (Dz2 - Dz1) / (B[0] - A[0]) * (vxmin + hhLength - A[0])
        V1 = [vxmin, min(localdpt,localdpt1) - 10]
        V2 = [vxmax, min(localdpt,localdpt1) - 10]
        V3 = [max(v3x, v4x), min(localdpt,localdpt1) - 10 + random.uniform(0.7, 1.3)*hhThickness]
        V4 = [min(v3x, v4x), min(localdpt,localdpt1) - 10 + random.uniform(0.7, 1.3)*hhThickness]
        mask1 = generatePolygon(xElementLocation, zElementLocation, V1, V2, V3, V4)
        # hollow
        mask[mask1 == 1] =0
    if HH == 2:
        vxmin = random.uniform(xElementLocation[14], xElementLocation[-15]-hhLength)
        vxmax = vxmin + hhLength
        v3x = random.uniform(vxmin, vxmax)
        v4x = random.uniform(vxmin, vxmax)
        localdpt = Dz1 + (Dz2 - Dz1) / (B[0] - A[0]) * (vxmin - A[0])
        localdpt1 = Dz1 + (Dz2 - Dz1) / (B[0] - A[0]) * (vxmin + hhLength - A[0])
        V1 = [min(v3x, v4x), max(localdpt,localdpt1) + 10 - random.uniform(0.7, 1.3)*hhThickness]
        V2 = [max(v3x, v4x), max(localdpt,localdpt1) + 10 - random.uniform(0.7, 1.3)*hhThickness]
        V3 = [vxmax, max(localdpt,localdpt1) + 10]
        V4 = [vxmin, max(localdpt,localdpt1) + 10]
        mask1 = generatePolygon(xElementLocation, zElementLocation, V1, V2, V3, V4)
        # hump
        mask[mask1 == 1] = 1
    lgrho_layered[mask==1]=baseX

    return lgrho_layered

def GenerateBlockSample(xElementLocation, zElementLocation, Vertex,Thickness,Length,Rotation,backX,baseX):
    lgrho_layered = np.reshape(backX,(-1,1),order='f')
    A = Vertex
    B = [Vertex[0]+Length*math.cos(Rotation/180*math.pi),Vertex[1]+Length*math.sin(Rotation/180*math.pi)]
    C = [Vertex[0]+Length*math.cos(Rotation/180*math.pi)-Thickness*math.sin(Rotation/180*math.pi),
         Vertex[1]+Length*math.sin(Rotation/180*math.pi)+Thickness*math.cos(Rotation/180*math.pi)]
    D = [Vertex[0]-Thickness*math.sin(Rotation/180*math.pi),Vertex[1]+Thickness*math.cos(Rotation/180*math.pi)]
    mask = generatePolygon(xElementLocation, zElementLocation, A, B, C, D)
    lgrho_layered[mask==1] = baseX
    return lgrho_layered

def GenerateQuadorTriSample(xElementLocation, zElementLocation, Vertex,Thickness,Length,edge,backX,baseX):
    lgrho_layered = np.reshape(backX,(-1,1),order='f')
    # sigedianzuobiao
    # edge == 4 or 3
    a = random.uniform(0.4, 1)
    V1 = [random.uniform(0,1), 0]
    V2 = [1, random.uniform(0,a)]
    V3 = [random.uniform(0,1),a]
    if edge == 4:  # quad
        V4 = [0,random.uniform(0,a)]
    else:  # tri
        V4 = V3.copy()
    # xuanzhuanzhongxindian 1/2, a/2
    rot1 = random.uniform(0, 360)
    M = np.array([[math.cos(rot1), -math.sin(rot1)],[math.sin(rot1), math.cos(rot1)]])
    n1pots = np.array([[V1[0], V2[0],V3[0],V4[0]],[V1[1], V2[1],V3[1],V4[1]]])
    n2pts = np.dot(M,n1pots)
    xmin = n2pts[0,:].min()
    xmax = n2pts[0,:].max()
    zmin = n2pts[1,:].min()
    zmax = n2pts[1,:].max()
    # bianhuanzhi zhidingquyu
    Xmin = Vertex[0]
    Xmax = Vertex[0] + Length
    Zmin = Vertex[1]
    Zmax = Vertex[1] + Thickness
    n3pts = n2pts.copy()
    n3pts[0,:] = (n2pts[0,:]-xmin)*(Xmax - Xmin)/(xmax - xmin) + Xmin
    n3pts[1,:] = (n2pts[1,:]-zmin)*(Zmax - Zmin)/(zmax - zmin) + Zmin
    mask = generatePolygon(xElementLocation, zElementLocation,
                           [n3pts[0,0],n3pts[1,0]], [n3pts[0,1],n3pts[1,1]], [n3pts[0,2],n3pts[1,2]], [n3pts[0,3],n3pts[1,3]])
    lgrho_layered[mask==1] = baseX
    return lgrho_layered

def GenerateCircleSample(xElementLocation, zElementLocation, Vertex,Radius,backX,baseX):
    lgrho_layered = np.reshape(backX,(-1,1),order='f')
    mask = 0*lgrho_layered.copy()
    for ii in range(len(mask)):
        if((xElementLocation[ii] - Vertex[0])**2 + (zElementLocation[ii] - Vertex[1])**2 <= Radius**2):
            mask[ii] == 1
    lgrho_layered[mask==1] = baseX
    return lgrho_layered

def SetupGeneralTrainingSet2(TrainSet2, ii):
    VelBlocktmp = TrainSet2['VelBlocktmp']
    xElementLocationInv = TrainSet2['xElementLocationInv']
    zElementLocationInv = TrainSet2['zElementLocationInv']
    JointMean = TrainSet2['JointMean']
    JointSigma = TrainSet2['JointSigma']
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 10
    expandY = 10
    w1 = fspecial_gaussian(np.array([12, 12]), 20)
    w2 = fspecial_gaussian(np.array([12, 12]), 20)
    Background = 1
    Basement = 2
    ZIndexSample = random.randint(10,45)

    Dz1 = zElementLocationInv[ZIndexSample]
    Dz2 = zElementLocationInv[ZIndexSample] + 500
    backFieldRho = Background * np.ones((ZNumberInv, XNumberInv))
    tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, backFieldRho, Basement)

    # Basement = 3
    # Dz1 = zElementLocationInv(ZIndexSample + 5) %
    # Dz2 = zElementLocationInv(ZIndexSample + 5)
    # tmp2 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, tmp1, Basement)

    AbnormalC = random.randint(3,4)   # 4
    Vertex = [random.uniform(1000, 4000), random.uniform(200, 1000)]
    Thickness = random.uniform(300, 500)
    Length = random.uniform(2000, 5000)
    Rotation = random.uniform(-2, 2)
    label = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, tmp1,
                                AbnormalC)

    RhoBlock2 = np.zeros(np.size(label))
    VelBlock2 = np.zeros(np.size(label))

    R = np.random.multivariate_normal(JointMean[0,:], np.reshape(JointSigma[0,:], (2, 2)))
    history_rho_1 = max(R[0], 0.1)
    history_vel_1 = R[1]
    R = np.random.multivariate_normal(JointMean[1,:], np.reshape(JointSigma[1,:], (2, 2)))
    history_rho_2 = max(R[0], 0.1)
    history_vel_2 = R[1]
    R = np.random.multivariate_normal(JointMean[2,:], np.reshape(JointSigma[2,:], (2, 2)))
    history_rho_3 = max(R[0], 0.1)
    history_vel_3 = R[1]
    R = np.random.multivariate_normal(JointMean[3,:], np.reshape(JointSigma[3,:], (2, 2)))
    history_rho_4 = max(R[0], 0.1)
    history_vel_4 = R[1]

    RhoBlock2[list(np.where(label == 1)[0])] = history_rho_1
    VelBlock2[list(np.where(label == 1)[0])] = history_vel_1
    RhoBlock2[list(np.where(label == 2)[0])] = history_rho_2
    VelBlock2[list(np.where(label == 2)[0])] = history_vel_2
    RhoBlock2[list(np.where(label == 3)[0])] = history_rho_3
    VelBlock2[list(np.where(label == 3)[0])] = history_vel_3
    RhoBlock2[list(np.where(label == 4)[0])] = history_rho_4
    VelBlock2[list(np.where(label == 4)[0])] = history_vel_4

    # VelBlock2 = VelBlocktmp(:) + VelBlock2

    MatRhoBlock2 = np.reshape(RhoBlock2, (ZNumberInv, XNumberInv),order='f')
    VLayeredMatTemp = imageExpand(MatRhoBlock2, expandX, expandY)
    VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w1)
    RhoBlock2 = VLayeredMatTemp[expandY:-expandY, expandX: - expandX]
    logFieldRhoTrain = np.log10(np.reshape(RhoBlock2,(-1,1),order='f'))
# '''    % interpRhoTrainMat = interp2(xInv, yInv, np.reshape(RhoBlock2, ZNumberInv, XNumberInv), xMT, yMT, 'nearest',
#                                   RhoBlock2(1))
#     % interpRhoTrain(i,:) = log10(interpRhoTrainMat(:))
#     % [rho_tr(i,:), !, !, !] = MT2DFWD2(frequencyMT, interpRhoTrain(
#         i,:), RxMT, xElementLocationMT, zElementLocationMT, XNumberMT, ZNumberMT, rxIndexMT)'''
    MatVelBlock2 = np.reshape(VelBlock2, (ZNumberInv, XNumberInv),order='f')
    # % MatVelBlock2(end - 2: end,:) = 4000
    VLayeredMatTemp = imageExpand(MatVelBlock2, expandX, expandY)
    VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w2)
    VelBlock2 = VLayeredMatTemp[expandY: - expandY, expandX: - expandX]
    logFieldVelTrain = np.log10(1 / np.reshape(VelBlock2,(-1,1),order='f'))
    # % interpVelTrainMat = interp2(xInv, yInv, np.reshape(VelBlock2, ZNumberInv, XNumberInv), xSMC, ySMC, 'nearest')
    # % interpVelTrain(i,:)= log10(1. / interpVelTrainMat(:))
    # % [time_tr(i,:), !] = SMC2DFWDSLV2(interpVelTrain(
    #     i,:), SnNodeIndexSMC, RnNodeIndexSMC, disMat, sparseRow, sparseCollum, activeNodeNumMat, infoTable)
    # end
    history_rho = [history_rho_1,history_rho_2,history_rho_3,history_rho_4]
    history_vel = [history_vel_1,history_vel_2,history_vel_3,history_vel_4]
    return {'Rho':logFieldRhoTrain, 'Vel':logFieldVelTrain, 'his_r':history_rho, 'his_v':history_vel}


def SetupGeneralCaseExample1(TrainSet1, N1):
    xElementLocationInv = TrainSet1['xElementLocationInv']
    zElementLocationInv = TrainSet1['zElementLocationInv']
    JointMean = TrainSet1['JointMean']
    JointSigma = TrainSet1['JointSigma']
    VelBlocktmp = TrainSet1['VelBlocktmp']
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 10
    expandY = 10
# w1 = fspecial('gaussian', [8 8], 2)
# w2 = fspecial('gaussian', [8 8], 2)
# w1 = fspecial('gaussian', [5 5], 1)
# w2 = fspecial('gaussian', [5 5], 1)
    w1 = fspecial_gaussian(np.array([12, 12]), 20)
    w2 = fspecial_gaussian(np.array([12, 12]), 20)
    logFieldRhoTrain = np.zeros((N1,ZNumberInv*XNumberInv))
    logFieldVelTrain = np.zeros((N1,ZNumberInv*XNumberInv))
    # logFieldRhoTrain = np.zeros((N1,ZNumberInv*XNumberInv))


    Background = 1
    Basement = 2
    ZIndexSample = random.randint(10,45)
    # ZIndexSample = 15
    Dz1 = zElementLocationInv[ZIndexSample]*(1+random.uniform(-0.1, 0.1))
    Dz2 = zElementLocationInv[ZIndexSample]*(1+random.uniform(-0.1, 0.1))
    backFieldRho = Background * np.ones((ZNumberInv, XNumberInv))
    tmp1 =GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, backFieldRho, Basement)
    #
    # % Basement = 3
    # % Dz1 = zElementLocationInv(ZIndexSample + 5) %
    # % Dz2 = zElementLocationInv(ZIndexSample + 5)
    # % tmp2 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, tmp1, Basement)
    # %
    AbnormalC = random.randint(3,4) # 4
    # AbnormalC = 4

    Thickness = random.uniform(300, 500)
    Length = random.uniform(2000, 4000)
    Rotation = random.uniform(-3, 3)
    # Thickness = 300
    # Length = 4000

    # Vertex = [random.uniform(1000, 4000), random.uniform(200, 1000)]
    # Vertex = [2000, 1050]
    Vertex = [random.uniform(1000, 9000-Length), random.uniform(200, 1000)]

    label = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, tmp1,
                                AbnormalC)

    RhoBlock2 = np.zeros(np.size(label))
    VelBlock2 = np.zeros(np.size(label))
    R = np.random.multivariate_normal(JointMean[0,:], np.reshape(JointSigma[0,:], (2, 2)))
    history_rho_1 = max(R[0], 0.1)
    history_vel_1 = R[1]
    R = np.random.multivariate_normal(JointMean[1,:], np.reshape(JointSigma[1,:], (2, 2)))
    history_rho_2 = max(R[0], 0.1)
    history_vel_2 = R[1]
    R = np.random.multivariate_normal(JointMean[2,:], np.reshape(JointSigma[2,:], (2, 2)))
    history_rho_3 = max(R[0], 0.1)
    history_vel_3 = R[1]
    R = np.random.multivariate_normal(JointMean[3,:], np.reshape(JointSigma[3,:], (2, 2)))
    history_rho_4 = max(R[0], 0.1)
    history_vel_4 = R[1]

    RhoBlock2[list(np.where(label==1)[0])] = history_rho_1
    VelBlock2[list(np.where(label==1)[0])] = history_vel_1
    RhoBlock2[list(np.where(label==2)[0])] = history_rho_2
    VelBlock2[list(np.where(label==2)[0])] = history_vel_2
    RhoBlock2[list(np.where(label==3)[0])] = history_rho_3
    VelBlock2[list(np.where(label==3)[0])] = history_vel_3
    RhoBlock2[list(np.where(label==4)[0])] = history_rho_4
    VelBlock2[list(np.where(label==4)[0])] = history_vel_4

    # RhoBlock2[list(np.where(label==1)[0])] = 30
    # VelBlock2[list(np.where(label==1)[0])] = 1500
    # RhoBlock2[list(np.where(label==2)[0])] = 50
    # VelBlock2[list(np.where(label==2)[0])] = 2000
    # RhoBlock2[list(np.where(label==3)[0])] = 13
    # VelBlock2[list(np.where(label==3)[0])] = 800
    # RhoBlock2[list(np.where(label==4)[0])] = 100
    # VelBlock2[list(np.where(label==4)[0])] = 3000

    # VelBlock2 = np.reshape(VelBlocktmp,-1,order='f') + VelBlock2

    MatRhoBlock2 = np.reshape(RhoBlock2, (ZNumberInv, XNumberInv), order='f')
    VLayeredMatTemp = imageExpand(MatRhoBlock2, expandX, expandY)
    VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w1)
    RhoBlock2 = VLayeredMatTemp[expandY: - expandY, expandX: - expandX]
    logFieldRhoTrain = np.log10(np.reshape(RhoBlock2,(-1,1),order = 'f'))
    # % interpRhoTrainMat = interp2(xInv, yInv, np.reshape(RhoBlock2, ZNumberInv, XNumberInv), xMT, yMT, 'nearest',
    #                               RhoBlock2(1))
    # % interpRhoTrain(i,:) = log10(interpRhoTrainMat(:))
    # % [rho_tr(i,:), !, !, !] = MT2DFWD2(frequencyMT, interpRhoTrain(
    #     i,:), RxMT, xElementLocationMT, zElementLocationMT, XNumberMT, ZNumberMT, rxIndexMT)
    #
    MatVelBlock2 = np.reshape(VelBlock2, (ZNumberInv, XNumberInv), order='f')
    # % % MatVelBlock2(end - 2: end,:) = 4000
    VLayeredMatTemp = imageExpand(MatVelBlock2, expandX, expandY)
    VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w2)
    VelBlock2 = VLayeredMatTemp[expandY: - expandY, expandX:  - expandX]
    logFieldVelTrain = np.log10(np.reshape(1 / VelBlock2, (-1,1), order='f'))
        # % interpVelTrainMat = interp2(xInv, yInv, np.reshape(VelBlock2, ZNumberInv, XNumberInv), xSMC, ySMC, 'nearest')
        # % interpVelTrain(i,:)= log10(1. / interpVelTrainMat(:))
        # % [time_tr(i,:), !] = SMC2DFWDSLV2(
        #     interpVelTrain(i,:), SnNodeIndexSMC, RnNodeIndexSMC, disMat, sparseRow, sparseCollum, activeNodeNumMat, infoTable)
        # end
    history_rho = [history_rho_1, history_rho_2, history_rho_3, history_rho_4]
    history_vel = [history_vel_1, history_vel_2, history_vel_3, history_vel_4]
    return {'Rho': logFieldRhoTrain, 'Vel': logFieldVelTrain, 'his_r': history_rho, 'his_v': history_vel}

def SetupGeneralCaseExample2(TrainSet2, N1):
    # 弱映射下的联合训练（模仿joint inversion算例二）
    # 三种：简单层  简单块   简单层 复杂块   复杂层  简单块  复杂层 复杂块\
    # 暂时停用
    kind= random.randint(0,3)
    kind1 = random.randint(0,2)  #
    xElementLocationInv = TrainSet2['xElementLocationInv']
    zElementLocationInv = TrainSet2['zElementLocationInv']
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 10
    expandY = 10
    w1 = fspecial_gaussian(np.array([12, 12]), 15)
    w2 = fspecial_gaussian(np.array([12, 12]), 15)
    Background = 1
    Basement = 2
    ZIndexSample1 = random.randint(20, 26)

    # layer 1, Layer 2  interface 250-350m
    Dz1a = zElementLocationInv[random.randint(12, 33)]
    Dz2a = zElementLocationInv[random.randint(12, 33)]
    backFieldRho = Background * np.ones((ZNumberInv, XNumberInv))
    if kind >=2 and kind1 == 0:
        tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1a, Dz2a, backFieldRho, Basement, random.randint(1,2), 2000, 200)
    else:
        tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1a, Dz2a, backFieldRho, Basement)

    # layer2, layer3 interface 1150-1250m
    Basement = 3
    ZIndexSample2 = random.randint(49, 51)
    Dz1bb = random.randint(45, 55)
    Dz2bb = random.randint(45, 55)
    Dz1b = zElementLocationInv[Dz1bb]  #
    Dz2b = zElementLocationInv[Dz2bb]
    if kind >= 2 and kind1 == 1:
        tmp2 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1b, Dz2b, tmp1, Basement, random.randint(1,2), 3000, 300)
    else:
        tmp2 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1b, Dz2b, tmp1, Basement)

    # layer3, layer4 interface 1400-1800m
    Basement = 4
    ZIndexSample3 = random.randint(54, 60)
    Dz1c = zElementLocationInv[random.randint(max(51,max(Dz1bb, Dz2bb)), 62)]
    Dz2c = zElementLocationInv[random.randint(max(51,max(Dz1bb, Dz2bb)), 62)]
    if kind >= 2 and kind1 == 2:
        tmp3 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1c, Dz2c, tmp2, Basement, random.randint(1,2), 4000, 300)
    else:
        tmp3 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1c, Dz2c, tmp2, Basement)

    # block1: in the first layer
    Abnormal = 5
    Thickness = random.uniform(100, 200)
    Length = random.uniform(2000, 3000)
    Vertex = [random.uniform(xElementLocationInv[14], xElementLocationInv[-15] - Length),
              random.uniform(0, max(Dz1a, Dz2a) - Thickness)]
    Rotation = random.uniform(-3, 3)
    if kind % 2 == 1:
        Thickness *= 1.3
        Length *= 1.3
        label1 = GenerateQuadorTriSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, random.randint(3,4), tmp3, Abnormal)
    else:
        label1 = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, tmp3, Abnormal)

    # block2: in the second layer
    Abnormal = 6
    Thickness = random.uniform(400, 600)
    Length = random.uniform(3000, 5000)
    Vertex = [random.uniform(xElementLocationInv[14], xElementLocationInv[-15] - Length),
              random.uniform(max(Dz1a, Dz2a), max(Dz1b, Dz2b) - Thickness)]
    Rotation = random.uniform(-3, 3)
    if kind % 2 == 1:
        Thickness *= 1.3
        Length *= 1.3
        label = GenerateQuadorTriSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, random.randint(3,4), label1, Abnormal)
    else:
        label = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, label1, Abnormal)

    # label = GenerateCircleSample(xElementLocation, zElementLocation, Vertex, Radius, backX, baseX)

    RhoBlock2 = np.zeros(np.size(label))
    VelBlock2 = np.zeros(np.size(label))
    # generate rho and vel   # distribution 1
    flg1 = random.randint(0, 1)
    if flg1 == 0:
        while (1):
            R = np.random.multivariate_normal(np.array([150, 1500]), np.array([[700, 0], [0, 50000]]))
            # R[1] = random.uniform(900, 2100)
            if (R[0] > 0 and R[1] > 0):
                break
    else:
        while (1):
            R = np.random.multivariate_normal(np.array([300, 1500]), np.array([[700, 100], [100, 90000]]))
            # R[1] = random.uniform(900,2100)
            if (R[0] > 0 and R[1] > 0):
                break
    history_rho_1 = R[0]
    history_vel_1 = R[1]
    flg1 = random.randint(0, 1)
    if flg1 == 0:
        while (1):
            R = np.random.multivariate_normal(np.array([150, 1500]), np.array([[700, 0], [0, 50000]]))
            # R[1] = random.uniform(1100, 1900)
            if (R[0] > 0 and R[1] > 0):
                break
    else:
        while (1):
            R = np.random.multivariate_normal(np.array([300, 1500]), np.array([[700, 100], [100, 90000]]))
            # R[1] = random.uniform(1000, 1950)
            if (R[0] > 0 and R[1] > 0):
                break
    history_rho_5 = R[0]
    history_vel_5 = R[1]

    # distribution 2
    flg2 = random.randint(0, 1)
    if flg2 == 0:
        while (1):
            R = np.random.multivariate_normal(np.array([160, 2400]), np.array([[700, 0], [0,70000]]))
            if (R[0] > 0 and R[1] > 0):
                break
    else:
        while (1):
            R = np.random.multivariate_normal(np.array([290, 2800]), np.array([[1000, 500], [500, 100000]]))
            if (R[0] > 0 and R[1] > 0):
                break
    history_rho_2 = R[0]
    history_vel_2 = R[1]
    flg2 = random.randint(0, 1)
    if flg2 == 0:
        while (1):
            R = np.random.multivariate_normal(np.array([50, 3800]), np.array([[400, -200], [-200, 40000]]))
            if (R[0] > 0 and R[1] > 0):
                break
    else:
        if random.randint(0,1) == 0:
            while (1):
                R = np.random.multivariate_normal(np.array([160, 2500]), np.array([[700, 0], [0, 70000]]))
                if (R[0] > 0 and R[1] > 0):
                    break
        else:
            while (1):
                R = np.random.multivariate_normal(np.array([290, 2900]), np.array([[1000, 500], [500, 90000]]))
                if (R[0] > 0 and R[1] > 0):
                    break
    history_rho_6 = R[0]
    history_vel_6 = R[1]

    # distribution 3
    while (1):
        R = np.random.multivariate_normal(np.array([310, 4100]), np.array([[1000, 300], [300, 90000]]))
        if (R[0] > 0 and R[1] > 0):
            break
    history_rho_3 = R[0]
    history_vel_3 = R[1]

    # distribution 4
    while (1):
        R = np.random.multivariate_normal(np.array([500, 5300]), np.array([[1000, 200], [200, 100000]]))
        if (R[0] > 0 and R[1] > 0):
            break
    history_rho_4 = R[0]
    history_vel_4 = R[1]

    RhoBlock2[list(np.where(label == 1)[0])] = history_rho_1
    VelBlock2[list(np.where(label == 1)[0])] = history_vel_1
    RhoBlock2[list(np.where(label == 2)[0])] = history_rho_2
    VelBlock2[list(np.where(label == 2)[0])] = history_vel_2
    RhoBlock2[list(np.where(label == 3)[0])] = history_rho_3
    VelBlock2[list(np.where(label == 3)[0])] = history_vel_3
    RhoBlock2[list(np.where(label == 4)[0])] = history_rho_4
    VelBlock2[list(np.where(label == 4)[0])] = history_vel_4
    RhoBlock2[list(np.where(label == 5)[0])] = history_rho_5
    VelBlock2[list(np.where(label == 5)[0])] = history_vel_5
    RhoBlock2[list(np.where(label == 6)[0])] = history_rho_6
    VelBlock2[list(np.where(label == 6)[0])] = history_vel_6

    # VelBlock2 = VelBlocktmp(:) + VelBlock2
    #
    # MatRhoBlock2 = np.reshape(RhoBlock2, (ZNumberInv, XNumberInv), order='f')
    # VLayeredMatTemp = imageExpand(MatRhoBlock2, expandX, expandY)
    # VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w1)
    # RhoBlock2 = VLayeredMatTemp[expandY:-expandY, expandX: - expandX]
    logFieldRhoTrain = np.log10(np.reshape(RhoBlock2, (-1, 1), order='f'))
    # '''    % interpRhoTrainMat = interp2(xInv, yInv, np.reshape(RhoBlock2, ZNumberInv, XNumberInv), xMT, yMT, 'nearest',
    #                                   RhoBlock2(1))
    #     % interpRhoTrain(i,:) = log10(interpRhoTrainMat(:))
    #     % [rho_tr(i,:), !, !, !] = MT2DFWD2(frequencyMT, interpRhoTrain(
    #         i,:), RxMT, xElementLocationMT, zElementLocationMT, XNumberMT, ZNumberMT, rxIndexMT)'''
    # MatVelBlock2 = np.reshape(VelBlock2, (ZNumberInv, XNumberInv), order='f')
    # #     # % MatVelBlock2(end - 2: end,:) = 4000
    # VLayeredMatTemp = imageExpand(MatVelBlock2, expandX, expandY)
    # VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w2)
    # VelBlock2 = VLayeredMatTemp[expandY: - expandY, expandX: - expandX]
    logFieldVelTrain = np.log10(1 / np.reshape(VelBlock2, (-1, 1), order='f'))
    # % interpVelTrainMat = interp2(xInv, yInv, np.reshape(VelBlock2, ZNumberInv, XNumberInv), xSMC, ySMC, 'nearest')
    # % interpVelTrain(i,:)= log10(1. / interpVelTrainMat(:))
    # % [time_tr(i,:), !] = SMC2DFWDSLV2(interpVelTrain(
    #     i,:), SnNodeIndexSMC, RnNodeIndexSMC, disMat, sparseRow, sparseCollum, activeNodeNumMat, infoTable)
    # end
    # vel : 慢度
    history_rho = [history_rho_1, history_rho_2, history_rho_3, history_rho_4, history_rho_5, history_rho_6]
    history_vel = [history_vel_1, history_vel_2, history_vel_3, history_vel_4, history_vel_5, history_vel_6]
    return {'Rho': logFieldRhoTrain, 'Vel': logFieldVelTrain, 'his_r': history_rho, 'his_v': history_vel}

def dist1():
    flg = random.randint(0,1)
    if flg == 0:
        while (1):
            R = np.random.multivariate_normal(np.array([150, 1200]), np.array([[1300, 1200], [1200, 60000]]))
            # R[1] = random.uniform(900, 2100)
            if (R[0] > 0 and R[1] > 0):
                break
    else:
        while (1):
            R = np.random.multivariate_normal(np.array([300, 1500]), np.array([[1500, 1200], [1200, 60000]]))
            # R[1] = random.uniform(900,2100)
            if (R[0] > 0 and R[1] > 0):
                break
    return R[0], R[1]

def dist1v(a,b):
    if a == 0:
        rho1 = 1200/np.sqrt(1300*60000)
        if b == 0:
            y = random.uniform(600,800)
                # R = np.random.multivariate_normal(np.array([150, 1200]), np.array([[1300, 1200], [1200, 60000]]))
                # R[1] = random.uniform(900, 2100)
        else:
            y = random.uniform(1600,1800)
        x = np.random.normal(150+np.sqrt(1300/60000)*rho1*(y-1200),1300*(1-rho1*rho1))
    else:
        rho1 = 1200/np.sqrt(1500*60000)
        if b == 0:
            y = random.uniform(900,1100)
        else:
            y = random.uniform(1900,2100)
            # R = np.random.multivariate_normal(np.array([300, 1500]), np.array([[1500, 1200], [1200, 60000]]))
            # R[1] = random.uniform(900,2100)
        x = np.random.normal(300+np.sqrt(1500/60000)*rho1*(y-1500),1500*(1-rho1*rho1))

    return x, y

def dist2v(a,b):
    if a == 0:
        rho1 = 1200/np.sqrt(1500*60000)
        if b == 0:
            y = random.uniform(1600,1800)
            # R = np.random.multivariate_normal(np.array([180, 2200]), np.array([[1500, 1200], [1200,60000]]))
            # R[1] = random.uniform(900, 2100)
        else:
            y = random.uniform(2600,2800)
        x = np.random.normal(180+np.sqrt(1500/60000)*rho1*(y-2200),1500*(1-rho1*rho1))
    else:
        rho1 = 1400/np.sqrt(1500*60000)
        if b == 0:
            y = random.uniform(2000,2200)
        else:
            y = random.uniform(3000,3200)
            # R = np.random.multivariate_normal(np.array([340, 2600]), np.array([[1500, 1400], [1400, 60000]]))
            # R[1] = random.uniform(900,2100)
        x = np.random.normal(340+np.sqrt(1500/60000)*rho1*(y-2600),1500*(1-rho1*rho1))

    return x, y

def dist2():
    flg = random.randint(0,1)
    if flg == 0:
        while (1):
            R = np.random.multivariate_normal(np.array([180, 2200]), np.array([[1500, 1200], [1200,60000]]))
            if (R[0] > 0 and R[1] > 0):
                break
    else:
        while (1):
            R = np.random.multivariate_normal(np.array([340, 2600]), np.array([[1500, 1400], [1400, 60000]]))
            if (R[0] > 0 and R[1] > 0):
                break
    return R[0], R[1]

def dist3():
    while (1):
        R = np.random.multivariate_normal(np.array([370, 3700]), np.array([[2200, 2800], [2800, 80000]]))
        if (R[0] > 0 and R[1] > 0):
            break
    return R[0], R[1]

def dist4():
    while (1):
        R = np.random.multivariate_normal(np.array([450, 4800]), np.array([[2700, 3400], [3400, 90000]]))
        if (R[0] > 0 and R[1] > 0):
            break
    return R[0], R[1]

def SetupGeneralCaseExample2c(TrainSet2, N1):
    # 弱映射下的联合训练（模仿joint inversion算例二）
    # 三种：简单层  简单块   简单层 复杂块   复杂层  简单块  复杂层 复杂块
    #  没有矿体异常
    kind= random.randint(0,3)
    kind1 = random.randint(0,2)  #
    xElementLocationInv = TrainSet2['xElementLocationInv']
    zElementLocationInv = TrainSet2['zElementLocationInv']
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 10
    expandY = 10
    w1 = fspecial_gaussian(np.array([12, 12]), 15)
    w2 = fspecial_gaussian(np.array([12, 12]), 15)
    Background = 1
    Basement = 2
    ZIndexSample1 = random.randint(20, 26)

    # layer 1, Layer 2  interface 250-350m
    Dz1a = zElementLocationInv[random.randint(12, 33)]
    Dz2a = zElementLocationInv[random.randint(12, 33)]
    backFieldRho = Background * np.ones((ZNumberInv, XNumberInv))
    if kind >=2 and kind1 == 0:
        tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1a, Dz2a, backFieldRho, Basement, random.randint(1,2), 2000, 200)
    else:
        tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1a, Dz2a, backFieldRho, Basement)

    # layer2, layer3 interface 1150-1250m
    Basement = 3
    ZIndexSample2 = random.randint(49, 51)
    Dz1bb = random.randint(45, 55)
    Dz2bb = random.randint(45, 55)
    Dz1b = zElementLocationInv[Dz1bb]  #
    Dz2b = zElementLocationInv[Dz2bb]
    if kind >= 2 and kind1 == 1:
        tmp2 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1b, Dz2b, tmp1, Basement, random.randint(1,2), 3000, 300)
    else:
        tmp2 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1b, Dz2b, tmp1, Basement)

    # layer3, layer4 interface 1400-1800m
    Basement = 4
    ZIndexSample3 = random.randint(54, 60)
    Dz1c = zElementLocationInv[random.randint(max(51,max(Dz1bb, Dz2bb)), 62)]
    Dz2c = zElementLocationInv[random.randint(max(51,max(Dz1bb, Dz2bb)), 62)]
    if kind >= 2 and kind1 == 2:
        tmp3 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1c, Dz2c, tmp2, Basement, random.randint(1,2), 4000, 300)
    else:
        tmp3 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1c, Dz2c, tmp2, Basement)

    # block1: in the first layer
    Abnormal = 5
    Thickness = random.uniform(100, 200)
    Length = random.uniform(2000, 3000)
    Vertex = [random.uniform(xElementLocationInv[14], xElementLocationInv[-15] - Length),
              random.uniform(0, max(Dz1a, Dz2a) - Thickness)]
    Rotation = random.uniform(-3, 3)
    if kind % 2 == 1:
        Thickness *= 1.3
        Length *= 1.3
        label1 = GenerateQuadorTriSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, random.randint(3,4), tmp3, Abnormal)
    else:
        label1 = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, tmp3, Abnormal)

    # block2: in the second layer
    Abnormal = 6
    Thickness = random.uniform(400, 600)
    Length = random.uniform(3000, 5000)
    Vertex = [random.uniform(xElementLocationInv[14], xElementLocationInv[-15] - Length),
              random.uniform(max(Dz1a, Dz2a), max(Dz1b, Dz2b) - Thickness)]
    Rotation = random.uniform(-3, 3)
    if kind % 2 == 1:
        Thickness *= 1.3
        Length *= 1.3
        label = GenerateQuadorTriSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, random.randint(3,4), label1, Abnormal)
    else:
        label = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, label1, Abnormal)

    # label = GenerateCircleSample(xElementLocation, zElementLocation, Vertex, Radius, backX, baseX)

    RhoBlock2 = np.zeros(np.size(label))
    VelBlock2 = np.zeros(np.size(label))
    # generate rho and vel   # distribution 1

    history_rho_1, history_vel_1 = dist1()

    history_rho_5, history_vel_5 = dist1()

    # distribution 2
    history_rho_2, history_vel_2 = dist2()
    history_rho_6, history_vel_6 = dist2()

    # distribution 3
    history_rho_3, history_vel_3 = dist3()

    history_rho_4, history_vel_4 = dist4()

    RhoBlock2[list(np.where(label == 1)[0])] = history_rho_1
    VelBlock2[list(np.where(label == 1)[0])] = history_vel_1
    RhoBlock2[list(np.where(label == 2)[0])] = history_rho_2
    VelBlock2[list(np.where(label == 2)[0])] = history_vel_2
    RhoBlock2[list(np.where(label == 3)[0])] = history_rho_3
    VelBlock2[list(np.where(label == 3)[0])] = history_vel_3
    RhoBlock2[list(np.where(label == 4)[0])] = history_rho_4
    VelBlock2[list(np.where(label == 4)[0])] = history_vel_4
    RhoBlock2[list(np.where(label == 5)[0])] = history_rho_5
    VelBlock2[list(np.where(label == 5)[0])] = history_vel_5
    RhoBlock2[list(np.where(label == 6)[0])] = history_rho_6
    VelBlock2[list(np.where(label == 6)[0])] = history_vel_6

    # VelBlock2 = VelBlocktmp(:) + VelBlock2
    #
    # MatRhoBlock2 = np.reshape(RhoBlock2, (ZNumberInv, XNumberInv), order='f')
    # VLayeredMatTemp = imageExpand(MatRhoBlock2, expandX, expandY)
    # VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w1)
    # RhoBlock2 = VLayeredMatTemp[expandY:-expandY, expandX: - expandX]
    logFieldRhoTrain = np.log10(np.reshape(RhoBlock2, (-1, 1), order='f'))
    # '''    % interpRhoTrainMat = interp2(xInv, yInv, np.reshape(RhoBlock2, ZNumberInv, XNumberInv), xMT, yMT, 'nearest',
    #                                   RhoBlock2(1))
    #     % interpRhoTrain(i,:) = log10(interpRhoTrainMat(:))
    #     % [rho_tr(i,:), !, !, !] = MT2DFWD2(frequencyMT, interpRhoTrain(
    #         i,:), RxMT, xElementLocationMT, zElementLocationMT, XNumberMT, ZNumberMT, rxIndexMT)'''
    # MatVelBlock2 = np.reshape(VelBlock2, (ZNumberInv, XNumberInv), order='f')
    # #     # % MatVelBlock2(end - 2: end,:) = 4000
    # VLayeredMatTemp = imageExpand(MatVelBlock2, expandX, expandY)
    # VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w2)
    # VelBlock2 = VLayeredMatTemp[expandY: - expandY, expandX: - expandX]
    logFieldVelTrain = np.log10(1 / np.reshape(VelBlock2, (-1, 1), order='f'))
    # % interpVelTrainMat = interp2(xInv, yInv, np.reshape(VelBlock2, ZNumberInv, XNumberInv), xSMC, ySMC, 'nearest')
    # % interpVelTrain(i,:)= log10(1. / interpVelTrainMat(:))
    # % [time_tr(i,:), !] = SMC2DFWDSLV2(interpVelTrain(
    #     i,:), SnNodeIndexSMC, RnNodeIndexSMC, disMat, sparseRow, sparseCollum, activeNodeNumMat, infoTable)
    # end
    # vel : 慢度
    history_rho = [history_rho_1, history_rho_2, history_rho_3, history_rho_4, history_rho_5, history_rho_6]
    history_vel = [history_vel_1, history_vel_2, history_vel_3, history_vel_4, history_vel_5, history_vel_6]
    return {'Rho': logFieldRhoTrain, 'Vel': logFieldVelTrain, 'his_r': history_rho, 'his_v': history_vel}


def SetupSpecificCaseExample1B(N1, VelBlocktmp, xElementLocationInv, zElementLocationInv, JointMean, JointSigma):
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 3
    expandY = 3
    # w1 = fspecial('gaussian', [8 8], 2)
    # w2 = fspecial('gaussian', [8 8], 2)
    # w1 = fspecial('gaussian', [5 5], 1)
    # w2 = fspecial('gaussian', [5 5], 1)
    w1 = fspecial_gaussian(np.array([12, 12]), 20)
    w2 = fspecial_gaussian(np.array([12, 12]), 20)
    logFieldRhoTrain = np.zeros((N1, ZNumberInv * XNumberInv))
    logFieldVelTrain = np.zeros((N1, ZNumberInv * XNumberInv))
    # logFieldRhoTrain = np.zeros((N1,ZNumberInv*XNumberInv))
    history_rho_1 = np.zeros(N1)
    history_rho_2 = np.zeros(N1)
    history_rho_3 = np.zeros(N1)
    history_rho_4 = np.zeros(N1)
    history_vel_1 = np.zeros(N1)
    history_vel_2 = np.zeros(N1)
    history_vel_3 = np.zeros(N1)
    history_vel_4 = np.zeros(N1)
    for i in range(N1):
        Background = 1
        Basement = 2
        ZIndexSample = random.randint(10, 45)
        # ZIndexSample = 15
        Dz1 = 500
        Dz2 = 500
        backFieldRho = Background * np.ones((ZNumberInv, XNumberInv))
        tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, backFieldRho, Basement)
        #
        # % Basement = 3
        # % Dz1 = zElementLocationInv(ZIndexSample + 5) %
        # % Dz2 = zElementLocationInv(ZIndexSample + 5)
        # % tmp2 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, tmp1, Basement)
        # %
        AbnormalC = random.randint(3, 4)  # 4
        AbnormalC = 3

        Thickness = random.uniform(300, 500)
        Length = random.uniform(2000, 4000)
        Rotation = random.uniform(-10, 10)
        Thickness = 300
        Length = 2500
        Rotation = -10

        Vertex = [random.uniform(1000, 9000 - Length), random.uniform(200, 1000)]
        Vertex = [5000, 1200]

        label = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, tmp1,
                                    AbnormalC)

        RhoBlock2 = np.zeros(np.size(label))
        VelBlock2 = np.zeros(np.size(label))
        R = np.random.multivariate_normal(JointMean[0, :], np.reshape(JointSigma[0, :], (2, 2)))
        history_rho_1[i] = max(R[0], 0.1)
        history_vel_1[i] = R[1]
        R = np.random.multivariate_normal(JointMean[1, :], np.reshape(JointSigma[1, :], (2, 2)))
        history_rho_2[i] = max(R[0], 0.1)
        history_vel_2[i] = R[1]
        R = np.random.multivariate_normal(JointMean[2, :], np.reshape(JointSigma[2, :], (2, 2)))
        history_rho_3[i] = max(R[0], 0.1)
        history_vel_3[i] = R[1]
        R = np.random.multivariate_normal(JointMean[3, :], np.reshape(JointSigma[3, :], (2, 2)))
        history_rho_4[i] = max(R[0], 0.1)
        history_vel_4[i] = R[1]

        RhoBlock2[list(np.where(label == 1)[0])] = history_rho_1[i]
        VelBlock2[list(np.where(label == 1)[0])] = history_vel_1[i]
        RhoBlock2[list(np.where(label == 2)[0])] = history_rho_2[i]
        VelBlock2[list(np.where(label == 2)[0])] = history_vel_2[i]
        RhoBlock2[list(np.where(label == 3)[0])] = history_rho_3[i]
        VelBlock2[list(np.where(label == 3)[0])] = history_vel_3[i]
        RhoBlock2[list(np.where(label == 4)[0])] = history_rho_4[i]
        VelBlock2[list(np.where(label == 4)[0])] = history_vel_4[i]

        RhoBlock2[list(np.where(label==1)[0])] = 30
        VelBlock2[list(np.where(label==1)[0])] = 1500
        RhoBlock2[list(np.where(label==2)[0])] = 50
        VelBlock2[list(np.where(label==2)[0])] = 2000
        RhoBlock2[list(np.where(label==3)[0])] = 10
        VelBlock2[list(np.where(label==3)[0])] = 800
        RhoBlock2[list(np.where(label==4)[0])] = 100
        VelBlock2[list(np.where(label==4)[0])] = 3000

        VelBlock2 = np.reshape(VelBlocktmp, -1, order='f') + VelBlock2

        # % MatRhoBlock2 = np.reshape(RhoBlock2, ZNumberInv, XNumberInv)
        # % VLayeredMatTemp = imageExpand(MatRhoBlock2, expandX, expandY)
        # % VLayeredMatTemp = imfilter(VLayeredMatTemp, w1)
        # % RhoBlock2 = VLayeredMatTemp(1 + expandY:end - expandY, 1 + expandX: end - expandX)
        logFieldRhoTrain[i, :] = np.log10(RhoBlock2)
        # % interpRhoTrainMat = interp2(xInv, yInv, np.reshape(RhoBlock2, ZNumberInv, XNumberInv), xMT, yMT, 'nearest',
        #                               RhoBlock2(1))
        # % interpRhoTrain(i,:) = log10(interpRhoTrainMat(:))
        # % [rho_tr(i,:), !, !, !] = MT2DFWD2(frequencyMT, interpRhoTrain(
        #     i,:), RxMT, xElementLocationMT, zElementLocationMT, XNumberMT, ZNumberMT, rxIndexMT)
        #
        # % MatVelBlock2 = np.reshape(VelBlock2, ZNumberInv, XNumberInv)
        # % % MatVelBlock2(end - 2: end,:) = 4000
        # % VLayeredMatTemp = imageExpand(MatVelBlock2, expandX, expandY)
        # % VLayeredMatTemp = imfilter(VLayeredMatTemp, w2)
        # % VelBlock2 = VLayeredMatTemp(1 + expandY:end - expandY, 1 + expandX: end - expandX)
        logFieldVelTrain[i, :] = np.log10(1 / VelBlock2)
        # % interpVelTrainMat = interp2(xInv, yInv, np.reshape(VelBlock2, ZNumberInv, XNumberInv), xSMC, ySMC, 'nearest')
        # % interpVelTrain(i,:)= log10(1. / interpVelTrainMat(:))
        # % [time_tr(i,:), !] = SMC2DFWDSLV2(
        #     interpVelTrain(i,:), SnNodeIndexSMC, RnNodeIndexSMC, disMat, sparseRow, sparseCollum, activeNodeNumMat, infoTable)
        # end
    history_rho = [history_rho_1, history_rho_2, history_rho_3, history_rho_4]
    history_vel = [history_vel_1, history_vel_2, history_vel_3, history_vel_4]
    return logFieldRhoTrain, logFieldVelTrain, history_rho, history_vel

def SetupSpecificCaseExample1C(N1, VelBlocktmp, xElementLocationInv, zElementLocationInv, JointMean, JointSigma):
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 3
    expandY = 3
    # w1 = fspecial('gaussian', [8 8], 2)
    # w2 = fspecial('gaussian', [8 8], 2)
    # w1 = fspecial('gaussian', [5, 5], 10)
    # w2 = fspecial('gaussian', [5, 5], 10)
    w1 = fspecial_gaussian(np.array([12, 12]), 20)
    w2 = fspecial_gaussian(np.array([12, 12]), 20)
    logFieldRhoTrain = np.zeros((N1, ZNumberInv * XNumberInv))
    logFieldVelTrain = np.zeros((N1, ZNumberInv * XNumberInv))
    # logFieldRhoTrain = np.zeros((N1,ZNumberInv*XNumberInv))
    history_rho_1 = np.zeros(N1)
    history_rho_2 = np.zeros(N1)
    history_rho_3 = np.zeros(N1)
    history_rho_4 = np.zeros(N1)
    history_vel_1 = np.zeros(N1)
    history_vel_2 = np.zeros(N1)
    history_vel_3 = np.zeros(N1)
    history_vel_4 = np.zeros(N1)
    for i in range(N1):
        Background = 1
        AbnormalC = 3
        # ZIndexSample = random.randint(10, 45)
        ZIndexSample = 36
        Dz1 = zElementLocationInv[ZIndexSample]
        Dz2 = zElementLocationInv[ZIndexSample]
        backFieldRho = Background * np.ones((ZNumberInv, XNumberInv))
        tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, backFieldRho, AbnormalC)

        Dz1 = zElementLocationInv[ZIndexSample+6]
        Dz2 = zElementLocationInv[ZIndexSample+6]
        Basement = 2
        tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, tmp1, Basement)

        AbnormalC = 4
        Vertex = [3000, 450]
        Thickness = 300
        Length = 4000
        Rotation = 0

        label = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, tmp1,AbnormalC)

        RhoBlock2 = np.zeros(np.size(label))
        VelBlock2 = np.zeros(np.size(label))
        R = np.random.multivariate_normal(JointMean[0, :], np.reshape(JointSigma[0, :], (2, 2)))
        history_rho_1[i] = max(R[0], 0.1)
        history_vel_1[i] = R[1]
        R = np.random.multivariate_normal(JointMean[1, :], np.reshape(JointSigma[1, :], (2, 2)))
        history_rho_2[i] = max(R[0], 0.1)
        history_vel_2[i] = R[1]
        R = np.random.multivariate_normal(JointMean[2, :], np.reshape(JointSigma[2, :], (2, 2)))
        history_rho_3[i] = max(R[0], 0.1)
        history_vel_3[i] = R[1]
        R = np.random.multivariate_normal(JointMean[3, :], np.reshape(JointSigma[3, :], (2, 2)))
        history_rho_4[i] = max(R[0], 0.1)
        history_vel_4[i] = R[1]

        RhoBlock2[list(np.where(label == 1)[0])] = history_rho_1[i]
        VelBlock2[list(np.where(label == 1)[0])] = history_vel_1[i]
        RhoBlock2[list(np.where(label == 2)[0])] = history_rho_2[i]
        VelBlock2[list(np.where(label == 2)[0])] = history_vel_2[i]
        RhoBlock2[list(np.where(label == 3)[0])] = history_rho_3[i]
        VelBlock2[list(np.where(label == 3)[0])] = history_vel_3[i]
        RhoBlock2[list(np.where(label == 4)[0])] = history_rho_4[i]
        VelBlock2[list(np.where(label == 4)[0])] = history_vel_4[i]

        RhoBlock2[list(np.where(label==1)[0])] = 30
        VelBlock2[list(np.where(label==1)[0])] = 1500
        RhoBlock2[list(np.where(label==2)[0])] = 50
        VelBlock2[list(np.where(label==2)[0])] = 2000
        RhoBlock2[list(np.where(label==3)[0])] = 10
        VelBlock2[list(np.where(label==3)[0])] = 800
        RhoBlock2[list(np.where(label==4)[0])] = 100
        VelBlock2[list(np.where(label==4)[0])] = 3000

        VelBlock2 = np.reshape(VelBlocktmp, -1, order='f') + VelBlock2

        # % MatRhoBlock2 = np.reshape(RhoBlock2, ZNumberInv, XNumberInv)
        # % VLayeredMatTemp = imageExpand(MatRhoBlock2, expandX, expandY)
        # % VLayeredMatTemp = imfilter(VLayeredMatTemp, w1)
        # % RhoBlock2 = VLayeredMatTemp(1 + expandY:end - expandY, 1 + expandX: end - expandX)
        logFieldRhoTrain[i, :] = np.log10(RhoBlock2)
        # % interpRhoTrainMat = interp2(xInv, yInv, np.reshape(RhoBlock2, ZNumberInv, XNumberInv), xMT, yMT, 'nearest',
        #                               RhoBlock2(1))
        # % interpRhoTrain(i,:) = log10(interpRhoTrainMat(:))
        # % [rho_tr(i,:), !, !, !] = MT2DFWD2(frequencyMT, interpRhoTrain(
        #     i,:), RxMT, xElementLocationMT, zElementLocationMT, XNumberMT, ZNumberMT, rxIndexMT)
        #
        # % MatVelBlock2 = np.reshape(VelBlock2, ZNumberInv, XNumberInv)
        # % % MatVelBlock2(end - 2: end,:) = 4000
        # % VLayeredMatTemp = imageExpand(MatVelBlock2, expandX, expandY)
        # % VLayeredMatTemp = imfilter(VLayeredMatTemp, w2)
        # % VelBlock2 = VLayeredMatTemp(1 + expandY:end - expandY, 1 + expandX: end - expandX)
        logFieldVelTrain[i, :] = np.log10(1 / VelBlock2)
        # % interpVelTrainMat = interp2(xInv, yInv, np.reshape(VelBlock2, ZNumberInv, XNumberInv), xSMC, ySMC, 'nearest')
        # % interpVelTrain(i,:)= log10(1. / interpVelTrainMat(:))
        # % [time_tr(i,:), !] = SMC2DFWDSLV2(
        #     interpVelTrain(i,:), SnNodeIndexSMC, RnNodeIndexSMC, disMat, sparseRow, sparseCollum, activeNodeNumMat, infoTable)
        # end
    history_rho = [history_rho_1, history_rho_2, history_rho_3, history_rho_4]
    history_vel = [history_vel_1, history_vel_2, history_vel_3, history_vel_4]
    return logFieldRhoTrain, logFieldVelTrain, history_rho, history_vel

def SetupQuasiNaturalModel01(N1, xElementLocationInv, zElementLocationInv):
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 3
    expandY = 3
    # w1 = fspecial('gaussian', [8 8], 2)
    # w2 = fspecial('gaussian', [8 8], 2)
    # w1 = fspecial('gaussian', [5 5], 1)
    # w2 = fspecial('gaussian', [5 5], 1)
    w1 = fspecial_gaussian(np.array([12, 12]), 20)
    w2 = fspecial_gaussian(np.array([12, 12]), 20)
    property = np.array([[30,5000], [1500,4800], [800,4000], [2000,3000], [100,2500]])
    logFieldRhoTrain = np.zeros((N1, ZNumberInv * XNumberInv))
    logFieldVelTrain = np.zeros((N1, ZNumberInv * XNumberInv))
    depth = np.tile(np.reshape(zElementLocationInv, (1, -1), order='f'),(1, len(xElementLocationInv)))
    depth = depth.squeeze()
    xmin = 0
    xmax = 10000
    zmin = zElementLocationInv.min()
    zmax = zElementLocationInv.max()
    for i in range(N1):
        label = 0
        # NumberBlock = random.randint(0,3)
        # NumberLayer = 3 - NumberBlock
        FieldRho = label * np.ones((ZNumberInv, XNumberInv))
        # 两层 2块 1楔形

        if random.randint(0,1) < 1:
            label = label + 1
            DzRange = [zmin + 0.4*(zmax-zmin), zmin + 0.6*(zmax-zmin)]
            Dz1 = random.uniform(DzRange[0], DzRange[1])
            Dz2 = random.uniform(DzRange[0], DzRange[1])
            FieldRho = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, FieldRho, label)
        if random.randint(0, 1) < 1:
            label = label + 1
            DzRange = [zmin + 0.6 * (zmax - zmin), zmin + 0.9 * (zmax - zmin)]
            Dz1 = random.uniform(DzRange[0], DzRange[1])
            Dz2 = random.uniform(DzRange[0], DzRange[1])
            FieldRho = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, FieldRho, label)
        if random.randint(0,2) < 2:
            label = label + 1
            vx = random.uniform(xmin, xmax)
            vz = random.uniform(zmin, zmin + 2/5*(zmax-zmin))
            Vertex = [vx, vz]
            Thickness = random.uniform(400, 900)
            Length = 20000
            Rotation = random.uniform(-25, 25)
            if vx < xmin + 1/2*(xmax - xmin):
                if random.randint(0,1) == 1:
                    Rotation = random.uniform(155, 180)
                else:
                    Rotation = random.uniform(-180, -155)
            FieldRho = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation,FieldRho, label)
        NumberBlock = random.randint(0,2)
        for tb in range(2):
            label = 8
            if random.randint(0,2) < 2:
                Rotation = random.uniform(-45, 45)
                Thickness = random.uniform(400, 600)
                Length = random.uniform(200, 3000)
                Vertex = [random.uniform(xmin + 1 / 5 * (xmax - xmin), xmin + 4 / 5 * (xmax - xmin) - Length),
                          random.uniform(zmin + 1 / 8 * (zmax - zmin), zmin + 2 / 5 * (zmax - zmin))]
                FieldRho = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation,FieldRho,label)


        RhoBlock2 = np.zeros(np.size(FieldRho))
        VelBlock2 = np.zeros(np.size(FieldRho))
        VelBlock2[list(np.where(FieldRho == 8)[0])] = property[0][1]
        RhoBlock2[list(np.where(FieldRho == 8)[0])] = property[0][0]
        property1 = property.copy()
        property1 = np.delete(property1, 0, 0)
        unq = np.unique(FieldRho)
        if unq[-1] == 8:
            unq = unq[:-1]
        unql = len(unq)
        depthth = np.zeros(unql)
        for tt in range(unql):
            l1 = depth[list(np.where(FieldRho == unq[tt])[0])]
            depthth[tt] = l1.sum()/len(l1)  # 10 300 200 100   0 3 2 1
        for tt in range(4-unql):
            excludeNum = random.randint(0, 3-tt)
            property1 = np.delete(property1, excludeNum, 0)
        weici = np.zeros(unql, dtype = 'int64')
        for tt in range(unql):
            weici[tt] = sum(sum([depthth > depthth[tt]]))
        for tt in range(unql):
            VelBlock2[list(np.where(FieldRho == unq[tt])[0])] = property1[weici[tt]][1]
            RhoBlock2[list(np.where(FieldRho == unq[tt])[0])] = property1[weici[tt]][0]

        # VelBlock2 = np.reshape(VelBlocktmp, -1, order='f') + VelBlock2
        logFieldRhoTrain[i, :] = np.log10(RhoBlock2)
        logFieldVelTrain[i, :] = np.log10(1 / VelBlock2)
    # history_rho = [history_rho_1, history_rho_2, history_rho_3, history_rho_4]
    # history_vel = [history_vel_1, history_vel_2, history_vel_3, history_vel_4]
    return logFieldRhoTrain, logFieldVelTrain

def SetupSpecificQuasiNaturalModel01(N1, xElementLocationInv, zElementLocationInv):
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 3
    expandY = 3
    # w1 = fspecial('gaussian', [8 8], 2)
    # w2 = fspecial('gaussian', [8 8], 2)
    # w1 = fspecial('gaussian', [5 5], 1)
    # w2 = fspecial('gaussian', [5 5], 1)
    w1 = fspecial_gaussian(np.array([12, 12]), 20)
    w2 = fspecial_gaussian(np.array([12, 12]), 20)
    property = np.array([[30,5000], [1500,4800], [800,4000], [2000,3000], [100,2500]])
    logFieldRhoTrain = np.zeros((N1, ZNumberInv * XNumberInv))
    logFieldVelTrain = np.zeros((N1, ZNumberInv * XNumberInv))
    depth = np.tile(np.reshape(zElementLocationInv, (1, -1), order='f'),(1, len(xElementLocationInv)))
    depth = depth.squeeze()
    xmin = 0
    xmax = 10000
    zmin = zElementLocationInv.min()
    zmax = zElementLocationInv.max()
    for i in range(N1):
        label = 0
        # NumberBlock = random.randint(0,3)
        # NumberLayer = 3 - NumberBlock
        FieldRho = label * np.ones((ZNumberInv, XNumberInv))
        # 两层 2块 1楔形

        if random.randint(0,1) < -1:
            label = label + 1
            DzRange = [zmin + 0.4*(zmax-zmin), zmin + 0.6*(zmax-zmin)]
            Dz1 = random.uniform(DzRange[0], DzRange[1])
            Dz2 = random.uniform(DzRange[0], DzRange[1])
            FieldRho = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, FieldRho, label)
        if random.randint(0, 1) < 2:
            label = label + 1
            DzRange = [zmin + 0.6 * (zmax - zmin), zmin + 0.8 * (zmax - zmin)]
            Dz1 = random.uniform(DzRange[0], DzRange[1])
            Dz2 = random.uniform(DzRange[0], DzRange[1])
            FieldRho = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, FieldRho, label)
        if random.randint(0,2) < -2:
            label = label + 1
            vx = random.uniform(xmin, xmax)
            vz = random.uniform(zmin, zmin + 2/5*(zmax-zmin))
            Vertex = [vx, vz]
            Thickness = random.uniform(400, 900)
            Length = 20000
            Rotation = random.uniform(-25, 25)
            if vx < xmin + 1/2*(xmax - xmin):
                if random.randint(0,1) == 1:
                    Rotation = random.uniform(155, 180)
                else:
                    Rotation = random.uniform(-180, -155)
            FieldRho = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation,FieldRho, label)
        NumberBlock = random.randint(0,2)
        for tb in range(1):
            label = 8
            if random.randint(0,2) < 3:
                Rotation = random.uniform(-15, 15)
                Thickness = random.uniform(300, 500)
                Length = random.uniform(1000, 3000)
                Vertex = [random.uniform(xmin + 2 / 5 * (xmax - xmin), xmin + 3 / 5 * (xmax - xmin) - Length),
                          random.uniform(zmin + 1 / 8 * (zmax - zmin), zmin + 1 / 5 * (zmax - zmin))]
                FieldRho = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation,FieldRho,label)


        RhoBlock2 = np.zeros(np.size(FieldRho))
        VelBlock2 = np.zeros(np.size(FieldRho))
        VelBlock2[list(np.where(FieldRho == 8)[0])] = property[0][1]
        RhoBlock2[list(np.where(FieldRho == 8)[0])] = property[0][0]
        property1 = property.copy()
        property1 = np.delete(property1, 0, 0)
        unq = np.unique(FieldRho)
        if unq[-1] == 8:
            unq = unq[:-1]
        unql = len(unq)
        depthth = np.zeros(unql)
        for tt in range(unql):
            l1 = depth[list(np.where(FieldRho == unq[tt])[0])]
            depthth[tt] = l1.sum()/len(l1)  # 10 300 200 100   0 3 2 1
        for tt in range(4-unql):
            excludeNum = random.randint(0, 3-tt)
            property1 = np.delete(property1, excludeNum, 0)
        weici = np.zeros(unql, dtype = 'int64')
        for tt in range(unql):
            weici[tt] = sum(sum([depthth > depthth[tt]]))
        for tt in range(unql):
            VelBlock2[list(np.where(FieldRho == unq[tt])[0])] = property1[weici[tt]][1]
            RhoBlock2[list(np.where(FieldRho == unq[tt])[0])] = property1[weici[tt]][0]

        # VelBlock2 = np.reshape(VelBlocktmp, -1, order='f') + VelBlock2
        logFieldRhoTrain[i, :] = np.log10(RhoBlock2)
        logFieldVelTrain[i, :] = np.log10(1 / VelBlock2)
    # history_rho = [history_rho_1, history_rho_2, history_rho_3, history_rho_4]
    # history_vel = [history_vel_1, history_vel_2, history_vel_3, history_vel_4]
    return logFieldRhoTrain, logFieldVelTrain

def SetupTrainingSetExample2(TrainSet2, ii):   # 复现论文过程的算例二
    # VelBlocktmp = TrainSet2['VelBlocktmp']
    xElementLocationInv = TrainSet2['xElementLocationInv']
    zElementLocationInv = TrainSet2['zElementLocationInv']
    ZNumberInv = len(zElementLocationInv)
    XNumberInv = len(xElementLocationInv)
    expandX = 10
    expandY = 10
    w1 = fspecial_gaussian(np.array([8, 8]), 2)
    w2 = fspecial_gaussian(np.array([8, 8]), 2)
    Background = 1
    Basement = 2
    ZIndexSample1 = random.randint(9, 10)

    # layer 1, Layer 2  interface 250-350m
    Dz1 = zElementLocationInv[ZIndexSample1]
    Dz2 = zElementLocationInv[ZIndexSample1]
    backFieldRho = Background * np.ones((ZNumberInv, XNumberInv))
    tmp1 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, backFieldRho, Basement)

    # layer2, layer3 interface 1150-1250m
    Basement = 3
    ZIndexSample2 = random.randint(38,40)
    Dz1 = zElementLocationInv[ZIndexSample2] #
    Dz2 = zElementLocationInv[ZIndexSample2]
    tmp2 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, tmp1, Basement)

    # layer3, layer4 interface 1400-1800m
    Basement = 4
    ZIndexSample3 = random.randint(42, 46)
    Dz1 = zElementLocationInv[ZIndexSample3]
    Dz2 = zElementLocationInv[ZIndexSample3]
    tmp3 = GenerateLayerSample(xElementLocationInv, zElementLocationInv, Dz1, Dz2, tmp2, Basement)

    # block1: in the first layer
    Abnormal = 5
    Thickness = random.uniform(100, 200)
    Length = random.uniform(3000, 6000)
    Vertex = [random.uniform(xElementLocationInv[0], xElementLocationInv[-1]-Length), random.uniform(0, 350-Thickness)]
    Rotation = random.uniform(-2, 2)
    label1 = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, tmp3,
                                Abnormal)

    # block2: in the second layer
    Abnormal = 6
    Thickness = random.uniform(300, 500)
    Length = random.uniform(3000, 6000)
    Vertex = [random.uniform(xElementLocationInv[0] + Length/2, xElementLocationInv[-1] - 3*Length/2), random.uniform(Vertex[1], 1250-Thickness)]
    Rotation = random.uniform(-2, 2)
    label = GenerateBlockSample(xElementLocationInv, zElementLocationInv, Vertex, Thickness, Length, Rotation, label1,
                                Abnormal)

    RhoBlock2 = np.zeros(np.size(label))
    VelBlock2 = np.zeros(np.size(label))
    # generate rho and vel
    while(1):
        R = np.random.multivariate_normal(np.array([110,1000]), np.array([[30, 100],[100, 800]]))
        if(R[0]>0 and R[0] < 100 and R[1]>0):
            break
    history_rho_1 = R[0]
    history_vel_1 = R[1]
    while(1):
        R = np.random.multivariate_normal(np.array([110,1000]), np.array([[30, 100],[100, 800]]))
        if(R[0]>50 and R[1]>0):
            break
    history_rho_5 = R[0]
    history_vel_5 = R[1]
    R[0] = random.uniform(12, 100)
    R[1] = -0.0533*R[0]**2 + 26.4*R[0] + 1141.3 + random.uniform(-200, 200)
    history_rho_2 = R[0]
    history_vel_2 = R[1]
    R[0] = random.uniform(80, 160)
    # R[1] = (np.log(R[0]-10))/(np.log(1.0019))+1400 + random.uniform(-200, 200)
    R[1] = -0.0533*R[0]**2 + 26.4*R[0] + 1141.3 +random.uniform(-200, 200)
    history_rho_6 = R[0]
    history_vel_6 = R[1]
    R[0] = random.uniform(15, 51)
    if(R[0] < 23):
        R[1] = 75*(R[0]-15)+3500 + random.uniform(-200,200)
    else:
        R[1] = 42.86*(R[0]-23)+4100 + random.uniform(-200,200)
    history_rho_3 = R[0]
    history_vel_3 = R[1]
    while(1):
        R = np.random.multivariate_normal(np.array([145, 5600]), np.array([[50,200],[200, 1200]]))
        if(R[0]>0 and R[1]>0):
            break
    history_rho_4 = R[0]
    history_vel_4 = R[1]

    RhoBlock2[list(np.where(label == 1)[0])] = history_rho_1
    VelBlock2[list(np.where(label == 1)[0])] = history_vel_1
    RhoBlock2[list(np.where(label == 2)[0])] = history_rho_2
    VelBlock2[list(np.where(label == 2)[0])] = history_vel_2
    RhoBlock2[list(np.where(label == 3)[0])] = history_rho_3
    VelBlock2[list(np.where(label == 3)[0])] = history_vel_3
    RhoBlock2[list(np.where(label == 4)[0])] = history_rho_4
    VelBlock2[list(np.where(label == 4)[0])] = history_vel_4
    RhoBlock2[list(np.where(label == 5)[0])] = history_rho_5
    VelBlock2[list(np.where(label == 5)[0])] = history_vel_5
    RhoBlock2[list(np.where(label == 6)[0])] = history_rho_6
    VelBlock2[list(np.where(label == 6)[0])] = history_vel_6

    # VelBlock2 = VelBlocktmp(:) + VelBlock2

    # MatRhoBlock2 = np.reshape(RhoBlock2, (ZNumberInv, XNumberInv), order='f')
    # VLayeredMatTemp = imageExpand(MatRhoBlock2, expandX, expandY)
    # VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w1)
    # RhoBlock2 = VLayeredMatTemp[expandY:-expandY, expandX: - expandX]
    logFieldRhoTrain = np.log10(np.reshape(RhoBlock2, (-1, 1), order='f'))
        # '''    % interpRhoTrainMat = interp2(xInv, yInv, np.reshape(RhoBlock2, ZNumberInv, XNumberInv), xMT, yMT, 'nearest',
        #                                   RhoBlock2(1))
        #     % interpRhoTrain(i,:) = log10(interpRhoTrainMat(:))
        #     % [rho_tr(i,:), !, !, !] = MT2DFWD2(frequencyMT, interpRhoTrain(
        #         i,:), RxMT, xElementLocationMT, zElementLocationMT, XNumberMT, ZNumberMT, rxIndexMT)'''
    # MatVelBlock2 = np.reshape(VelBlock2, (ZNumberInv, XNumberInv), order='f')
    #     # % MatVelBlock2(end - 2: end,:) = 4000
    # VLayeredMatTemp = imageExpand(MatVelBlock2, expandX, expandY)
    # VLayeredMatTemp = filters.convolve(VLayeredMatTemp, w2)
    # VelBlock2 = VLayeredMatTemp[expandY: - expandY, expandX: - expandX]
    logFieldVelTrain = np.log10(1 / np.reshape(VelBlock2, (-1, 1), order='f'))
    # % interpVelTrainMat = interp2(xInv, yInv, np.reshape(VelBlock2, ZNumberInv, XNumberInv), xSMC, ySMC, 'nearest')
    # % interpVelTrain(i,:)= log10(1. / interpVelTrainMat(:))
    # % [time_tr(i,:), !] = SMC2DFWDSLV2(interpVelTrain(
    #     i,:), SnNodeIndexSMC, RnNodeIndexSMC, disMat, sparseRow, sparseCollum, activeNodeNumMat, infoTable)
    # end
    # vel : 慢度
    history_rho = [history_rho_1, history_rho_2, history_rho_3, history_rho_4, history_rho_5, history_rho_6]
    history_vel = [history_vel_1, history_vel_2, history_vel_3, history_vel_4, history_vel_5, history_vel_6]
    return {'Rho': logFieldRhoTrain, 'Vel': logFieldVelTrain, 'his_r': history_rho, 'his_v': history_vel}

    # [verticalGradientSMC, horizontalGradientSMC] = computeGradient(xElementLocationSMC, zElementLocationSMC, XNumberSMC,
    #                                                                ZNumberSMC)
def computeGradient(Field_grid_x,Field_grid_z,X_number,Z_number):
    gridNumber = X_number * Z_number
    iVertical = -1 * np.ones(2*gridNumber)
    jVertical = -1 * np.ones(2*gridNumber)
    valueVertical = -10 * np.ones(2*gridNumber)
    iHorizontal = -1 * np.ones(2*gridNumber)
    jHorizontal = -1 * np.ones(2*gridNumber)
    valueHorizontal = -10 * np.ones(2*gridNumber)
    indexVertical = 0
    indexHorizontal = 0
    for i in range(gridNumber):
        temp1 = i % Z_number
        if temp1 != Z_number-1:
            iVertical[indexVertical] = i
            jVertical[indexVertical] = i

        # % valueVertical[indexVertical] = 1 / deltaZ(temp1)
            valueVertical[indexVertical] = 1
            indexVertical = indexVertical + 1
    
            iVertical[indexVertical] = i
            jVertical[indexVertical] = i + 1
            # % valueVertical[indexVertical] = -1 / deltaZ(temp1)
            valueVertical[indexVertical] = -1
            indexVertical = 1 + indexVertical
        # end
        temp2 = math.ceil(i / Z_number)
        if i < Z_number * (X_number - 1):
            iHorizontal[indexHorizontal] = i
            jHorizontal[indexHorizontal] = i
        # % valueHorizontal[indexHorizontal] = 1 / deltaX(temp2)
            valueHorizontal[indexHorizontal] = 1
            indexHorizontal = 1 + indexHorizontal

            iHorizontal[indexHorizontal] = i
            jHorizontal[indexHorizontal] = i + Z_number
            # % valueHorizontal[indexHorizontal] = -1 / deltaX(temp2)
            valueHorizontal[indexHorizontal] = -1
            indexHorizontal = 1 + indexHorizontal

    iVertical = iVertical[iVertical != -1]
    jVertical = jVertical[jVertical != -1]
    valueVertical = valueVertical[valueVertical!= -10]
    iHorizontal = iHorizontal[iHorizontal!= -1]
    jHorizontal = jHorizontal[jHorizontal!= -1]
    valueHorizontal = valueHorizontal[valueHorizontal!= -10]
    verticalGradient = csr_matrix((valueVertical,(iVertical, jVertical)) , shape = (gridNumber, gridNumber))
    horizontalGradient = csr_matrix((valueHorizontal,(iHorizontal, jHorizontal)), shape = (gridNumber, gridNumber))
    return verticalGradient, horizontalGradient

# class RV_dataset(torch.utils.data.Dataset):
#         """Class for getting individual transformations and data
#         Args:
#             images_dir = path of input images
#             labels_dir = path of labeled images
#             transformI = Input Images transformation (default: None)
#             transformM = Input Labels transformation (default: None)
#         Output:
#             tx = Transformed images
#             lx = Transformed labels"""
#
#         def __init__(self, inputs, outputs):
#             self.inputs = inputs
#             self.outputs = outputs
#
#         def __len__(self):
#
#             return len(self.inputs)
#
#         def __getitem__(self, i):
#             i1 = self.inputs[:, :, :, i]
#             l1 = self.outputs[:, :, i]
#
#             return i1, l1


# class P2T_dataset(torch.utils.data.Dataset):
#     """
#     other_ch=1:有其他multiinput
#     other_ch=-0：没有其他multiinput
#     """
#
#     def __init__(self, inputs, outputs, other_ch, depth):
#         self.inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
#         self.outputs = torch.from_numpy(outputs).type(torch.FloatTensor)
#         self.other_ch = other_ch
#         if other_ch == 1:
#             self.depth = depth
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, i):
#
#         i1 = self.inputs[:, :, :, i]
#         l1 = self.outputs[:, :, :, i]
#         if self.other_ch == 1:
#             i1 = [i1, self.depth[:,:,:,i]]
#
#         return i1, l1

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

'''
def cal_loss(y_pred, y):
    delta_y = y_pred-y
    acc = 0
    for j in range(np.size(delta_y, 0)):
        acc = acc + torch.sum(delta_y[j, :, :]**2)/torch.sum(y[j,:,:]**2)/np.size(delta_y, 0)
    loss = torch.sum(delta_y**2)/(np.size(delta_y,0)*np.size(delta_y,1)*np.size(delta_y,2))
    return loss, acc
'''
'''
class MylossFunc(nn.Module):
    def __int__(self):
        super(MylossFunc, self).__init__()

    def forward(self, net_output, desire_output):
        misfit = 0
        loss = torch.mean(torch.pow((net_output - desire_output), 2))
        # delta_y = net_output - desire_output
        # for j in range(delta_y.size(0)):
        #     for kk in range(delta_y.size(1)):
        #         misfit = misfit + torch.sum(delta_y[j, kk, :, :] ** 2) / torch.sum(desire_output[j, kk, :, :] ** 2) / delta_y.size(
        #             0) / delta_y.size(1)
        return loss
'''
# def cal_loss_multiple_layer(y_pred, y):
#     delta_y = y_pred-y
#     misfit = 0
#     for j in range(np.size(delta_y, 0)):
#         for kk in range(np.size(delta_y, 1)):
#             misfit = misfit + torch.sum(delta_y[j, kk, :, :]**2)/torch.sum(y[j,kk, :,:]**2)/np.size(delta_y, 0)/np.size(delta_y, 1)
#     loss = torch.sum(delta_y**2)/(np.size(delta_y,0)*np.size(delta_y,1)*np.size(delta_y,2)*np.size(delta_y, 3))
#     return loss, misfit

def Plot2DImage(domainDistance, domainDepth, xElementLocation, zElementLocation, X, PLOT_FLAG, colorBarAxis, contourStep,
            SURF_FALG, address, name, rangex=[0, 10], rangez=[-2,0], if_get_exp = 0, if_jet='jet'):
    # 画一张图
    ZNumber = len(zElementLocation)
    XNumber = len(xElementLocation)
    [x, y] = np.meshgrid(xElementLocation / 1000, -zElementLocation / 1000)
    if 'mt' == PLOT_FLAG:
        if if_get_exp==0:
            a = np.reshape(X, (ZNumber, XNumber),order = 'f')
        else:
            a = np.reshape(10**X, (ZNumber, XNumber), order='f')
        if SURF_FALG == 1:
            plt.ion()
            fig = plt.figure(figsize=(8,3))
            ax1 = fig.add_subplot(1,1,1)
            # fig1.axes[0].cla()
            # im = plt.imshow(a, cmap=plt.get_cmap('rainbow'), interpolation = 'nearest',extent = [x.min(), x.max(), y.min(), y.max()])
            plt.pcolor(x, y, a, cmap=plt.get_cmap(if_jet))
            # plt.title('Field Resistivity')
            # plt.draw()
            # plt.show()
            plt.xlim(rangex[0], rangex[1])
            plt.ylim(rangez[0], rangez[1])
            # fig1.axes[0].set_ylim(-domainDepth / 1000, 0)
            plt.xlabel('Distance (km)')
            plt.ylabel('Depth (km)')
            # ax1.view_init(azim = 0, elev = 90)
            # view(0, 90)
            cbar = plt.colorbar()
            if if_get_exp==0:
                plt.clim(colorBarAxis[0], colorBarAxis[1])
            else:
                plt.clim(10**colorBarAxis[0], 10**colorBarAxis[1])

            # set(gca, 'FontSize', 12)    # 设置坐标轴标注的字体
            # h = colorbar
            # cbar.set_label(r'Resistivity ($\Omega$m)')
            cbar.set_label('Logarithm of Resistivity')
            plt.tight_layout()
            plt.savefig(address + name)
            # plt.savefig(address + '/resistivity_start.png', bbox_inches='tight', pad_inches=0.0)
            # plt.savefig('./Result/1.png')
            plt.ioff()
            plt.close()
            # plt.show()
            # plt.pause(0.001)

    elif 'tt' == PLOT_FLAG:
        slownessMatFig = np.reshape(10 ** (X), (ZNumber, XNumber), order='f')
        if SURF_FALG == 1:
            plt.ion()
            plt.figure(figsize=(10,2))
            plt.subplot(1, 1, 1)
            plt.pcolor(x, y, 1/slownessMatFig, cmap=plt.get_cmap('jet'))
            # plt.imshow(1/slownessMatFig, cmap=plt.get_cmap('rainbow'), interpolation='nearest', extent = [x.min(), x.max(), y.min(), y.max()])
            # plt.title('Field Velocity')
            # plt.draw()
            # plt.show()
            plt.xlim(rangex[0], rangex[1])
            plt.ylim(rangez[0], rangez[1])
            plt.xlabel('Distance (km)')
            plt.ylabel('Depth (km)')
            cbar = plt.colorbar()
            plt.clim(colorBarAxis[0], colorBarAxis[1])
            # set(gca, 'FontSize', 12)    # 设置坐标轴标注的字体
            # h = colorbar
            cbar.set_label('Velocity (m/s)')
            plt.tight_layout()
            plt.ioff()
            plt.savefig(address + name)
            # plt.savefig(address + '/velocity_start.png', bbox_inches='tight', padding_inches=0.0)
            plt.close()
            # plt.show()
            # set(h, 'FontSize', 12)
            # set(gcf, 'Position', [500 500 498 198])
            # plt.pause(0.001)

    elif 'post-stack' == PLOT_FLAG:
        slownessMatFig = np.reshape((X), (ZNumber, XNumber), order='f')
        if SURF_FALG == 1:
            plt.figure(figsize=(10,2))
            plt.subplot(1, 1, 1)
            plt.pcolor(x, y, slownessMatFig, cmap=plt.get_cmap('seismic'))
            # plt.imshow(1/slownessMatFig, cmap=plt.get_cmap('rainbow'), interpolation='nearest', extent = [x.min(), x.max(), y.min(), y.max()])
            # plt.title('Field Velocity')
            # plt.draw()
            # plt.show()
            plt.xlim(rangex[0], rangex[1])
            plt.ylim(rangez[0], rangez[1])
            plt.xlabel('Distance (km)')
            plt.ylabel('Depth (km)')
            cbar = plt.colorbar()
            plt.clim(colorBarAxis[0], colorBarAxis[1])
            # set(gca, 'FontSize', 12)    # 设置坐标轴标注的字体
            # h = colorbar
            cbar.set_label('post-stack data')
            plt.tight_layout()
            plt.savefig(address + name)
            # plt.savefig(address + '/velocity_start.png', bbox_inches='tight', padding_inches=0.0)
            plt.close()


def PlotComparison2Image(xElementLocationMT, zElementLocationMT,ZNumberMT, XNumberMT,
                         xElementLocationSMC, zElementLocationSMC, ZNumberSMC, XNumberSMC,
                         newLogResist, LogRhoRef, newLogSlowness,LogSlownessRef,
                         title11, title12, title21,title22, addr1, addr2,colorBarAxis1, colorBarAxis2):
    # 画四张图，2*2，两种物理场模型对比
    [x, y] = np.meshgrid(xElementLocationMT / 1000, -zElementLocationMT / 1000)
    [x1, y1] = np.meshgrid(xElementLocationSMC / 1000, -zElementLocationSMC / 1000)
    plt.figure(figsize=(15,4))
    ax1 = plt.subplot(2,2, 1)
    _field1 = np.reshape(10**newLogResist, (ZNumberMT, XNumberMT), order='f')
    im1 = ax1.pcolor(x, y, _field1, cmap=plt.get_cmap('rainbow'))
    # im1 = ax1.imshow(_field1, cmap=plt.get_cmap('rainbow'), interpolation='bicubic', extent = [x.min(), x.max(), y.min(), y.max()])
    ax1.set_title(title11)
    ax1.set_xlim([0, 10])
    ax1.set_ylim([-2, 0])
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im1, ax=ax1)
    cbar.set_label(r'$Resistivity (\Omega m)$')
    im1.set_clim(10**colorBarAxis1[0], 10**colorBarAxis1[1])

    ax1 = plt.subplot(2, 2, 2)
    ax1.set_title(title21)
    newLogSlownessMat = 10 ** np.reshape(newLogSlowness, (ZNumberSMC, XNumberSMC), order='f')
    im1 = ax1.pcolor(x1, y1, 1 / newLogSlownessMat, cmap=plt.get_cmap('rainbow'))
    # im1 = ax1.imshow(1 / newLogSlownessMat, cmap=plt.get_cmap('rainbow'), interpolation='bicubic',extent = [x.min(), x.max(), y.min(), y.max()])
    ax1.set_xlim([0, 10000 / 1000])
    ax1.set_ylim([-2000 / 1000, 0])
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im1, ax=ax1)
    cbar.set_label(r'$Velocity (m/s)$')
    im1.set_clim(colorBarAxis2[0], colorBarAxis2[1])


    ax1 = plt.subplot(2,2,3)
    _field2 = np.reshape(10**LogRhoRef, (ZNumberMT, XNumberMT), order='f')
    im1 = ax1.pcolor(x, y, _field2, cmap=plt.get_cmap('rainbow'))
    # im1 = ax1.imshow(_field2, cmap=plt.get_cmap('rainbow'), interpolation='bicubic',extent = [x.min(), x.max(), y.min(), y.max()])
    ax1.set_title(title12)
    ax1.set_xlim([0, 10])
    ax1.set_ylim([-2, 0])
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    # cbaxes = fig1.add_axes([0.85, 0.1, 0.05, 0.8])
    cbar = plt.colorbar(im1, ax=ax1)
    im1.set_clim([10**colorBarAxis1[0], 10**colorBarAxis1[1]])
    cbar.set_label(r'$Resistivity (\Omega m)$')

    ax2 = plt.subplot(224)
    ax2.set_title(title22)
    newLogSlownessMat = 10 ** np.reshape(LogSlownessRef, (ZNumberSMC, XNumberSMC), order='f')
    im1 = ax2.pcolor(x1, y1, 1 / newLogSlownessMat, cmap=plt.get_cmap('rainbow'))
    # im1 = ax2.imshow(1 / newLogSlownessMat, cmap=plt.get_cmap('rainbow'), interpolation='bicubic',extent = [x.min(), x.max(), y.min(), y.max()])
    ax2.set_xlim([0, 10])
    ax2.set_ylim([-2, 0])
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im1, ax=ax2)
    cbar.set_label(r'$Velocity (m/s)$')
    im1.set_clim(colorBarAxis2[0], colorBarAxis2[1])

    plt.tight_layout()
    # plt.draw()
    # plt.show()
    plt.ioff()
    plt.savefig(addr1 + '.png',padding_inches=0.0)
    # plt.show()
    plt.close()

def PlotComparison1Image_v0(xElementLocationMT, zElementLocationMT, ZNumberMT, XNumberMT,
                         newLogResist, LogRhoRef, title11, title21, addr1, colorBarAxis1):
    # 画两张图，2*1，一种物理场模型对比
    [x, y] = np.meshgrid(xElementLocationMT / 1000, -zElementLocationMT / 1000)
    plt.figure(figsize=(7.5, 4))
    ax1 = plt.subplot(2, 1, 1)
    _field1 = np.reshape(10**newLogResist, (ZNumberMT, XNumberMT), order='f')
    im1 = ax1.pcolor(x, y, _field1, cmap=plt.get_cmap('jet'))
    # im1 = ax1.imshow(_field1, cmap=plt.get_cmap('jet'), interpolation='bicubic', extent = [x.min(), x.max(), y.min(), y.max()])
    ax1.set_title(title11)
    ax1.set_xlim([0, 10])
    ax1.set_ylim([-2, 0])
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im1, ax=ax1)
    cbar.set_label(r'$Resistivity (\Omega m)$')
    im1.set_clim(10**colorBarAxis1[0], 10**colorBarAxis1[1])
    # plt.clim(1, 200)

    ax1 = plt.subplot(2, 1, 2)
    _field1 = np.reshape(10**LogRhoRef, (ZNumberMT, XNumberMT), order='f')
    im1 = ax1.pcolor(x, y, _field1, cmap=plt.get_cmap('jet'))
    # im1 = ax1.imshow(_field1, cmap=plt.get_cmap('jet'), interpolation='bicubic', extent = [x.min(), x.max(), y.min(), y.max()])
    ax1.set_title(title21)
    ax1.set_xlim([0, 10])
    ax1.set_ylim([-2, 0])
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im1, ax=ax1)
    cbar.set_label(r'$Resistivity (\Omega m)$')
    im1.set_clim(10**colorBarAxis1[0], 10**colorBarAxis1[1])

    plt.tight_layout()
    # plt.draw()
    # plt.show()
    plt.ioff()
    plt.savefig(addr1 + '.png', padding_inches=0.0)
    # plt.show()
    plt.close()

def PlotComparison1Image(xElementLocationMT, zElementLocationMT, ZNumberMT, XNumberMT,
                         newLogResist, LogRhoRef, title11, title21, addr1, colorBarAxis1, rangex=[0,10], rangez=[-2,0]
                         , method1='contourf', method2 = 'contourf'):
    # 画两张图，2*1，一种物理场模型对比
    [x, y] = np.meshgrid(xElementLocationMT / 1000, -zElementLocationMT / 1000)
    plt.figure(figsize=(7.5, 4))
    ax1 = plt.subplot(2, 1, 1)
    _field1 = np.reshape(newLogResist, (ZNumberMT, XNumberMT), order='f')
    if method1 == 'contourf':
        im1 = ax1.contourf(x, y, _field1, 20,cmap=plt.get_cmap('jet'))
    else:
        im1 = ax1.pcolor(x, y, _field1,cmap=plt.get_cmap('jet'))
    # im1 = ax1.imshow(_field1, cmap=plt.get_cmap('rainbow'), interpolation='bicubic', extent = [x.min(), x.max(), y.min(), y.max()])
    ax1.set_title(title11)
    ax1.set_xlim(rangex)
    ax1.set_ylim(rangez)
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im1, ax=ax1)
    cbar.set_label(r'$Resistivity (\Omega m)$')
    im1.set_clim(colorBarAxis1[0], colorBarAxis1[1])
    # plt.clim(1, 200)

    ax1 = plt.subplot(2, 1, 2)
    _field1 = np.reshape(LogRhoRef, (ZNumberMT, XNumberMT), order='f')
    if method2 == 'contourf':
        im1 = ax1.contourf(x, y, _field1, 20, cmap=plt.get_cmap('jet'))
    else:
        im1 = ax1.pcolor(x, y, _field1, cmap=plt.get_cmap('jet'))
    # im1 = ax1.imshow(_field1, cmap=plt.get_cmap('rainbow'), interpolation='bicubic', extent = [x.min(), x.max(), y.min(), y.max()])
    ax1.set_title(title21)
    ax1.set_xlim(rangex)
    ax1.set_ylim(rangez)
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im1, ax=ax1)
    cbar.set_label(r'$Resistivity (\Omega m)$')
    im1.set_clim(colorBarAxis1[0], colorBarAxis1[1])

    plt.tight_layout()
    # plt.draw()
    # plt.show()
    plt.ioff()
    plt.savefig(addr1 + '.png', padding_inches=0.0)
    # plt.show()
    plt.close()

def model_misfit_cal(m0,m1,m2,m3):
    mm1 = lg.norm(np.reshape((m1 - m0), -1, order='f')) / lg.norm(
        np.reshape(m0, -1, order='f'))
    mm2 = lg.norm(np.reshape((m2 - m0), -1, order='f')) / lg.norm(
        np.reshape(m0, -1, order='f'))
    mm3 = lg.norm(np.reshape((m3 - m0), -1, order='f')) / lg.norm(
        np.reshape(m0, -1, order='f'))
    return mm1, mm2, mm3

def mf_cal(base, target):
    mf = lg.norm(np.reshape(target, -1, order='f')-np.reshape(base,-1,order='f'))/lg.norm(np.reshape(base,-1,order='f'))
    return mf

def zor(matr):
    maax = np.max(matr)
    miin = np.min(matr)
    return (matr-miin)/(maax-miin)


def get_model(style_layers_names, layer_num, group_num, ZNumber, XNumber):
    """
    :param content_layers_names: content layers names list
    :param style_layers_names: style layers name list
    :return: neural transfer model
    """
    # vgg = tf.train.load_checkpoint()
    vgg = VGG19(include_top=True, weights=None)  # 删掉最后一层；默认加载ImageNet上的预训练权重
    vgg.load_weights('VGG19/vgg19.h5')
    vgg._set_dtype_policy(tf.float64) ### 2023年9月6日添加：用双精度浮点数计算
    # vgg.load_weights('S:/我的资料库/Seismic_style MT/VGG19/vgg19.h5')
    # vgg.load_weights('./VGG19/vgg_19.ckpt')

    layer1_wb = vgg.layers[1].get_weights()

    layer1_w = layer1_wb[0]
    layer1_w_newshape = list(layer1_w.shape)
    layer1_w_newshape[2] = 1
    layer1_w_newshape = tuple(layer1_w_newshape)

    layer1_w_new = np.zeros(layer1_w_newshape)
    layer1_w_new[:, :, 0, :] = np.mean(layer1_w, axis=2)

    layer1_wb_new = layer1_wb
    layer1_wb_new[0] = layer1_w_new

    config = vgg.get_layer('block1_conv1').get_config()
    #     layer1_new = Conv2D(config['filters'], kernel_size=config['kernel_size'], \
    # strides=config['strides'], input_shape=(224, 224, 1), \
    # padding=config['padding'], data_format=config['data_format'], \
    # activation=config['activation'], use_bias=config['use_bias'],weights=layer1_wb_new)
    layer1_new = Conv2D(64, (3, 3), activation='relu', padding=config['padding'], use_bias=True, name='block1_conv1')
    input_shape = tf.TensorShape([None, ZNumber, XNumber, 1])  # to define h, w, c based on shape of layer input
    layer1_new.build(input_shape)
    layer1_new.set_weights(layer1_wb_new)
    neural_transfer_models = []
    for jj in range(len(style_layers_names)):
        input_new = Input(shape=(ZNumber, XNumber, 1))

        layers = [l for l in vgg.layers]

        x = layer1_new(input_new)
        if (style_layers_names[jj] != 'block1_conv1'):  ######可能会根据实际使用的层名调整
            for i in range(2, len(layers)):
                x = layers[i](x)
                if (layers[i].name == style_layers_names[jj]):
                    break

        vgg_new = Model(input_new, x)

        vgg_new.trainable = False  # 参数不可训练

        for kk in range(group_num):
            style_output = [vgg_new.output]
            style_output1 = []
            A = style_output[0]
            style_output1.append(A)
            # get the content layer and style layer
            # content_output = [vgg.get_layer(name=layer).output for layer in content_layers_names]
            # style_output = [vgg.get_layer(name=layer).output for layer in style_layers_names]
            # ####
            # style_output1 = []
            # A = style_output[0]
            # style_output1.append(A[:,:,:,:layer_num])
            # ####
            model_output = style_output1  # list combine

            # get the neural transfer model
            neural_transfer_model = Model(input_new, model_output)
            neural_transfer_models.append(neural_transfer_model)
    return neural_transfer_models

def gen_fine2coar_matrix(xPSFiTe, yPSFiTe, xMT, yMT, xElementLocationMT,
                         zElementLocationMT, xElementLocationPSFiTe, zElementLocationPSFiTe):
    [ZNumberPSFiTe, XNumberPSFiTe] = np.shape(xPSFiTe)
    [ZNumberMT, XNumberMT] = np.shape(xMT)
    fine_2_coar = np.zeros((XNumberPSFiTe * ZNumberPSFiTe, XNumberMT * ZNumberMT))
    xPSs = np.reshape(xPSFiTe, -1, order='f')
    yPSs = np.reshape(yPSFiTe, -1, order='f')
    xMTs = np.reshape(xMT, -1, order='f')
    yMTs = np.reshape(yMT, -1, order='f')
    ias = []
    jas = []
    indices = []
    vals = []
    for pp in range(XNumberPSFiTe * ZNumberPSFiTe):
        ias.append(pp)
        d_min = 1e9
        dd_min = -1
        print(pp)
        xpp = xPSs[pp]
        ypp = -yPSs[pp]
        ddx = np.argmin(abs(xpp - xElementLocationMT))
        ddy = np.argmin(abs(ypp - zElementLocationMT))
        dd_min = ddx * ZNumberMT + ddy
        # print("Method1, dd_min {}".format(dd_min))
        # for dd in range(XNumberMT*ZNumberMT):
        #     if(np.sqrt((xPSs[pp]-xMTs[dd])**2+(yPSs[pp]-yMTs[dd])**2) < d_min):
        #         d_min = np.sqrt((xPSs[pp]-xMTs[dd])**2+(yPSs[pp]-yMTs[dd])**2)
        #         dd_min = dd
        # print("Method2, dd_min {}".format(dd_min))
        vals.append(1)
        jas.append(dd_min)
        indices.append([pp, dd_min])

    # indices = tf.sparse.reorder(indices)
    # fine_2_coar = tf.sparse.SparseTensor(indices, vals, [XNumberPSFiTe*ZNumberPSFiTe, XNumberMT*ZNumberMT])
    ias = np.arange(0, ZNumberPSFiTe * XNumberPSFiTe)
    ias = ias.tolist()
    fine_2_coar_sparse = csr_matrix((vals, (ias, jas)), shape=(XNumberPSFiTe * ZNumberPSFiTe, XNumberMT * ZNumberMT))
    ###
    # fine_2_coar_dense = tf.sparse.to_dense(fine_2_coar)
    fine_2_coar_dense = np.array(fine_2_coar_sparse.todense(), dtype=np.float32)
    for pp in range(XNumberMT * ZNumberMT):
        if (np.sum(fine_2_coar_dense[:, pp]) != 0):
            # print(np.sum(fine_2_coar_dense[:, pp]))
            fine_2_coar_dense[:, pp] = fine_2_coar_dense[:, pp] / np.sum(fine_2_coar_dense[:, pp])
        else:
            print("Coarse No = {}, nearest interp".format(pp))
            xpp = xMTs[pp]
            ypp = -yMTs[pp]
            ddx = np.argmin(abs(xpp - xElementLocationPSFiTe))
            ddy = np.argmin(abs(ypp - zElementLocationPSFiTe))
            dd_min = ddx * ZNumberPSFiTe + ddy
            fine_2_coar_dense[dd_min, pp] = 1

    fine_2_coar_sparse = csr_matrix(fine_2_coar_dense)
    ###
    savemat('fine_2_coar-{}{}-{}{}_spa.mat'.format(ZNumberPSFiTe, XNumberPSFiTe, ZNumberMT, XNumberMT),
            {'fine_2_coar_sparse': fine_2_coar_sparse})
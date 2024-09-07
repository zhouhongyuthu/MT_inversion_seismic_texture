import math
import numpy as np
import time
from scipy.ndimage import filters
import random
# from pathosmultiprocessing import ProcessingPoll as Pool
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
# import networkx as nx
from scipy.io import savemat
import tensorflow as tf
from tensorflow.keras import backend as K

def computeCrossGradientAndJacobian(Field_rho_vec,vel_vec,Field_grid_x,Field_grid_z):
    Z_number = len(Field_grid_z)
    X_number = len(Field_grid_x)
    gridNumber = Z_number * X_number
    Field_rho_mat = np.reshape(Field_rho_vec, (Z_number, X_number), order = 'f')
    vel_mat = np.reshape(vel_vec, (Z_number, X_number), order='f')
    fieldRho = np.zeros((Z_number + 1, X_number + 1))
    fieldRho[0: Z_number, 0: X_number] = Field_rho_mat
    fieldRho[-1,:] = fieldRho[-2 ,:]
    fieldRho[:, -1] = fieldRho[:, -2]
    fieldVec = np.zeros((Z_number + 1, X_number + 1))
    fieldVec[0: Z_number, 0: X_number] = vel_mat
    fieldVec[-1,:] = fieldVec[-2,:]
    fieldVec[:, -1] = fieldVec[:, -2]
    partialRhoPartialZ = fieldRho[1:, 0: - 1] - fieldRho[0: - 1, 0: - 1]
    partialRhoPartialX = fieldRho[0:- 1, 1:] - fieldRho[0: - 1, 0: - 1]
    partialVecPartialZ = fieldVec[1:, 0: - 1] - fieldVec[0: - 1, 0: - 1]
    partialVecPartialX = fieldVec[0:- 1, 1:] - fieldVec[0: - 1, 0: - 1]

    deltaZ = np.zeros((Z_number,X_number))
    deltaX = np.zeros((Z_number, X_number))
    [deltaX[:-1,:-1], deltaZ[:-1,:-1]] = np.meshgrid(np.diff(Field_grid_x), np.diff(Field_grid_z))
    deltaX[-1,:] = deltaX[-2,:]
    deltaX[:, -1] = deltaX[:, -2]
    deltaZ[-1,:] = deltaZ[-2,:]
    deltaZ[:, -1] = deltaZ[:, -2]
    deltaX = np.ones((np.size(deltaX,0), np.size(deltaX,1)))
    deltaZ = np.ones((np.size(deltaZ,0), np.size(deltaZ,1)))
    crossGradientMatrix = (partialRhoPartialZ * partialVecPartialX - partialRhoPartialX * partialVecPartialZ) / deltaX / deltaZ

    # compute cross gradient Jacobian
    sparseVectorIndex = 0
    gridIndex = 0
    ia = -90 * np.ones(60000)
    ia = ia.astype(np.int64)
    ja = ia.copy()
    value = ia.copy()
    value = value.astype(np.float64)
    # % 同时写出rho和s的雅可比矩阵 将t rho s当做向量写之后转置（向量竖式）
    for i in range(X_number):
        for j in range(Z_number):
            mtIndex = gridIndex
            seismicIndex = mtIndex + X_number * Z_number

            ia[sparseVectorIndex] = mtIndex
            ja[sparseVectorIndex] = gridIndex
            value[sparseVectorIndex] = (fieldVec[j + 1, i] - fieldVec[j, i + 1]) / deltaX[j, i] / deltaZ[j, i]
            sparseVectorIndex = sparseVectorIndex + 1

            if (j + 1 <= Z_number-1):
                ia[sparseVectorIndex] = mtIndex + 1
                ja[sparseVectorIndex] = gridIndex
                value[sparseVectorIndex] = (fieldVec[j, i + 1] - fieldVec[j, i]) / deltaX[j, i] / deltaZ[j, i]
                sparseVectorIndex = sparseVectorIndex + 1


            if (i + 1 <= X_number-1):
                ia[sparseVectorIndex] = mtIndex + Z_number
                ja[sparseVectorIndex] = gridIndex
                value[sparseVectorIndex] = (fieldVec[j, i] - fieldVec[j + 1, i]) / deltaX[j, i] / deltaZ[j, i]
                sparseVectorIndex = sparseVectorIndex + 1


            ia[sparseVectorIndex] = seismicIndex
            ja[sparseVectorIndex] = gridIndex
            value[sparseVectorIndex] = (fieldRho[j, i + 1] - fieldRho[j + 1, i]) / deltaX[j, i] / deltaZ[j, i]
            sparseVectorIndex = sparseVectorIndex + 1

            if (j + 1 <= Z_number-1):
                ia[sparseVectorIndex] = seismicIndex + 1
                ja[sparseVectorIndex] = gridIndex
                value[sparseVectorIndex] = (fieldRho[j, i] - fieldRho[j, i + 1]) / deltaX[j, i] / deltaZ[j, i]
                sparseVectorIndex = sparseVectorIndex + 1


            if (i + 1 <= X_number-1):
                ia[sparseVectorIndex] = seismicIndex + Z_number
                ja[sparseVectorIndex] = gridIndex
                value[sparseVectorIndex] = (fieldRho[j + 1, i] - fieldRho[j, i]) / deltaX[j, i] / deltaZ[j, i]
                sparseVectorIndex = sparseVectorIndex + 1

            gridIndex = gridIndex + 1


    ia = ia[ia != -90]
    ja = ja[ja != -90]
    value = value[value != -90]
    crossGradientJacobian = csr_matrix((value, (ia, ja)))
    jacobianRho = crossGradientJacobian[0:gridNumber,:].T
    jacobianVel = crossGradientJacobian[gridNumber:,:].T
    return crossGradientMatrix, jacobianRho,jacobianVel


def ComputeJacobianFunc_z(EobsVector, HobsVector, EFieldVectorf, frequencyMT, LogField_rho, Rx,
                          Field_grid_x, Field_grid_z, X_number, Z_number, rxIndexMT,ia_temp, ja, value,Domain, index1):
    freqNumberMT = len(frequencyMT)
    rxNumberMT = len(rxIndexMT)
    Field_sigma = 1 / 10 ** LogField_rho
    gridNumber = X_number * Z_number
    dataLengthMT = rxNumberMT* freqNumberMT
    Rx_z = Rx[0]
    Rx_x = Rx[1]
    Grid_air_num = 7
    eachlenz = Field_grid_z[1] - Field_grid_z[0]
    Grid_z1 = eachlenz * np.arange(-1 - Grid_air_num, 0)
    Optimal_grid_number_z2 = 70
    temp2 = np.logspace(math.log10(Field_grid_z[-1] - Field_grid_z[-2]) + 0.1, 5, Optimal_grid_number_z2)
    Optimal_grid_z2 = np.cumsum(np.append(temp2, temp2[-1])) + Field_grid_z[-1]
    Grid_location_z = np.append(Grid_z1 + Field_grid_z[0], Field_grid_z)
    Grid_location_z = np.append(Grid_location_z, Optimal_grid_z2)
    Air = Grid_location_z[Grid_location_z < 0]
    Surface = np.argmax(Air)
    Grid_location_x = np.append(2 * Field_grid_x[0] - Field_grid_x[1], Field_grid_x)
    Grid_location_x = np.append(Grid_location_x, 2 * Field_grid_x[-1] - Field_grid_x[-2])
    Nx = len(Grid_location_x)
    Nz = len(Grid_location_z)
    Nxz = (Nx - 2) * (Nz - 2)
    J = 1
    Epsilon = 1 / 36 / math.pi * 1e-9
    Mu = 4 * math.pi * 1e-7
    Field_sigma_temp = np.reshape(Field_sigma, (Z_number, X_number), order='f')
    Back_sigma = Field_sigma_temp[-1, :]
    Sigma = np.zeros((Nz, Nx))
    Sigma[Grid_location_z > Field_grid_z[-1], :] = np.tile(
        np.append(np.append(Back_sigma[0], Back_sigma), Back_sigma[-1]),
        (Optimal_grid_number_z2 + 1, 1))
    Sigma[Surface + 1: Surface + 1 + Z_number, 1:-1] = Field_sigma_temp
    Sigma[Grid_location_z < 0, :] = 1e-6
    deltaX = np.diff(Grid_location_x[1:])
    deltaZ = np.diff(Grid_location_z[1:])
    deltaXArray = np.zeros((Nz - 2, Nx - 2))
    deltaZArray = np.zeros((Nz - 2, Nx - 2))
    for j in range(Nz-2):
        for i in range(Nx-2):
            deltaXArray[j, i] = deltaX[i]
            deltaZArray[j, i] = deltaZ[j]

    deltaXVector = np.reshape(deltaXArray, -1,order='f')
    deltaZVector = np.reshape(deltaZArray, -1,order='f')
    meshAreaVector = deltaXVector * deltaZVector
    meshAreaArray = deltaXArray * deltaZArray

    meshFieldAreaArray = meshAreaArray[Surface - 1:Surface - 1 + Z_number,:]
    meshFieldAreaVector = np.reshape(meshFieldAreaArray, -1,order='f')
    meshAreaArray = np.reshape(meshAreaArray, -1, order='f')
    EsErjErkFieldVec = np.zeros((gridNumber, 2 * rxNumberMT, freqNumberMT), dtype='complex')
    firstZLocation = Grid_location_z[0]
    Rxz_index = math.ceil((Rx_z - firstZLocation) / eachlenz)

    Sigma_active = np.reshape(Sigma[1:- 1, 1: - 1], -1,order='f')
    for gr in range(freqNumberMT):
        Omega = 2 * math.pi * frequencyMT[gr]
        value[index1:] = value[index1:] + -1j * Omega * Mu * Sigma_active
        value[index1:] = value[index1:] + Omega ** 2 * Mu * Epsilon
        D = csr_matrix((value, (ia_temp, ja)))
        value[index1:] = value[index1:] - -1j * Omega * Mu * Sigma_active
        value[index1:] = value[index1:] - Omega ** 2 * Mu * Epsilon
        UPointJCurrent = np.zeros(((Nz - 2) * X_number, rxNumberMT), dtype='complex')
        UPointKCurrent = np.zeros(((Nz - 2) * X_number, rxNumberMT), dtype='complex')
        for j in range(rxNumberMT):
            UPointJCurrent[int(Domain[Rxz_index, int(rxIndexMT[j])]-1), j] = -1j * Omega * Mu * (-1) / \
                                                                             meshAreaArray[int(Domain[Rxz_index, int(rxIndexMT[j])]-1)]
            UPointKCurrent[int(Domain[Rxz_index - 1, int(rxIndexMT[j])]-1), j] = 1 / 2 / \
                                                                                 (deltaZVector[int(Domain[Rxz_index, int(rxIndexMT[j])]-1)]) / meshAreaArray[int(Domain[Rxz_index, int(rxIndexMT[j])]-1)]
            # % 除以2是因为用了中心差分
            UPointKCurrent[int(Domain[Rxz_index + 1, int(rxIndexMT[j])]-1), j] = -1 / 2 / \
                                                                                 (deltaZVector[int(Domain[Rxz_index + 1, int(rxIndexMT[j])]-1)]) / meshAreaArray[int(Domain[Rxz_index, int(rxIndexMT[j])]-1)]


        URight = np.concatenate((UPointJCurrent, UPointKCurrent),axis=1)
        Ux = linalg.spsolve(D, URight)
        UxMatrix = np.reshape(Ux, (Nz - 2, Nx - 2, np.size(URight, 1)), order='f')
        ExFieldGrid = UxMatrix[Surface - 1:Surface - 1 + Z_number,:,:]
        EsErjErkFieldVec[:,:, gr]=np.reshape(ExFieldGrid, (gridNumber, np.size(URight, 1)),order='f')
        # inspect7 = {"UR": URight, "ExF":ExFieldGrid, "EsE":EsErjErkFieldVec}
        # savemat("inspect7.mat", inspect7)

    partialEPartialSigmaArray = np.zeros((gridNumber, rxNumberMT, freqNumberMT),dtype='complex')
    partialHPartialSigmaArray = np.zeros((gridNumber, rxNumberMT, freqNumberMT),dtype='complex')
    EFieldVectorReshape = np.reshape(EFieldVectorf, (gridNumber, 1, freqNumberMT),order='f')
    tempMeshFieldAreaVector = np.tile(np.reshape(meshFieldAreaVector,(-1,1,1),order='f'), (1, 1, freqNumberMT))
    for i in range(rxNumberMT):
        partialEPartialSigmaArray[:, i,:] = EFieldVectorReshape.squeeze()* EsErjErkFieldVec[:, i,:]
        partialEPartialSigmaArray[:, i,:] = partialEPartialSigmaArray[:, i,:]*tempMeshFieldAreaVector.squeeze()
        partialHPartialSigmaArray[:, i,:] = \
            EFieldVectorReshape.squeeze() * EsErjErkFieldVec[:, len(rxIndexMT) + i,:]*tempMeshFieldAreaVector.squeeze()

    partialEPartialSigma = np.reshape(partialEPartialSigmaArray, (gridNumber, dataLengthMT),order='f')
    partialHPartialSigma = np.reshape(partialHPartialSigmaArray, (gridNumber, dataLengthMT), order='f')
    # % appResistivityRepmat = repmat(10. ^ (Logrho_f), gridNumber, 1)
    # % appResistivityRepmat = repmat((Logrho_f), gridNumber, 1)
    EobsVectorRepmat = np.tile(EobsVector, (gridNumber, 1))
    HobsVectorRepmat = np.tile(HobsVector, (gridNumber, 1))
    partialAppResistivityPartialSigma = 2 * (np.real(partialEPartialSigma / EobsVectorRepmat) -
                                             np.real(partialHPartialSigma / HobsVectorRepmat))
    FieldRhoRepmat = np.tile(10** np.reshape(LogField_rho,(-1,1),order='f'), (1, dataLengthMT))
    # % FieldRhoRepmat = repmat(LogField_rho, 1, dataLengthMT)
    jacobianRho = -partialAppResistivityPartialSigma / FieldRhoRepmat
    jacobianPhi = -(np.log(10) / FieldRhoRepmat * (np.imag(partialEPartialSigma / EobsVectorRepmat)
                                                   - np.imag(partialHPartialSigma / HobsVectorRepmat)))
    return jacobianRho.T, jacobianPhi.T

def ComputeSlownessJacobianV2(lgSLNS,disMat,pathAll,NSx,NSz,Sn,Rn,activeNodeNumMat,infoTable):
    slowness=10**(lgSLNS)
    sparselen = 0
    # pathAll total length in 76*16 cells sum up
    for i in range(len(Sn)):
        for j in range(len(Rn)):
            sparselen = sparselen + len(pathAll[i][j])
    colNum = -1 * np.ones(sparselen)
    rowNum = -1*np.ones(sparselen)
    value = -1*np.ones(sparselen)  #长度超出初始值  python考虑初始化问题
    index = 0
    # % clear elementNumber distance
    for ps in range(len(Sn)):
        for pr in range(len(Rn)):
            path=pathAll[ps][pr]
            for i in range(len(path)-1):
                firstNode=path[i]
                secondNode=path[i+1]
                grd = np.zeros(4)
                aw = infoTable[int(firstNode)-1, 4: 6]
                grd[0:2] = aw
                grd[2: ] = infoTable[int(secondNode)-1, 4: 6]
                u,c = np.unique(grd, return_counts=True)
                colNum[index] = max(u[c>1])-1    #% Slowness Found to compute Jacobian
                # in 4 corners will return two numbers(0, the other), get the larger one
                rowNum[index]=ps*len(Rn)+pr #% 16个76，每个源-漏对
                value[index]=disMat[int(firstNode)-1,int(secondNode)-1] * slowness[int(colNum[index])] * np.log(10)
                # % t_j = SIGMA(slow_i*dis_i), partialt_j
                # % partialz_i,z_i=log(slow_i)
                # %             value(index)=disMat(firstNode,secondNode)
                index=index+1

    rowNum=rowNum[rowNum!=-1]
    colNum=colNum[colNum!=-1]
    value=value[0:len(colNum)]
    J=csr_matrix((value,(rowNum,colNum)),(len(Sn)*len(Rn),NSx*NSz))
    return J

def ComputeJacobianFuncTM2(EobsVector,HobsVector,ExFieldVector,EzFieldVector,frequencyMT,LogField_rho,Rx,Field_grid_x, Field_grid_z, X_number,Z_number,rxIndexMT):
    freqNumberMT=len(frequencyMT)
    rxNumberMT=len(rxIndexMT)
    # % Back_sigma = 1./10.^(LogBack_rho)
    Field_sigma = 1/10**LogField_rho
    # % Field_sigma = 1./LogField_rho

    gridNumber=X_number*Z_number
    dataLengthMT=rxNumberMT*freqNumberMT

    Rx_z = Rx[0]
    Grid_air_num=7

    eachlenz=Field_grid_z[1]-Field_grid_z[0]
    Grid_z1 = eachlenz * np.arange(-1 - Grid_air_num, 0)
    # %     Grid_z1 = []
    Optimal_grid_number_z2 = 30
    temp2 = np.logspace(math.log10(Field_grid_z[-1]-Field_grid_z[-2])+0.1,6,Optimal_grid_number_z2)
    Optimal_grid_z2 = np.cumsum(np.append(temp2,temp2[-1]))+Field_grid_z[-1]
    Grid_location_z = np.append(Grid_z1+Field_grid_z[0],Field_grid_z)
    Grid_location_z = np.append(Grid_location_z, Optimal_grid_z2)
    Air = Grid_location_z[Grid_location_z<5]
    Surface=np.argmax(Air)
    Grid_location_x = np.append(2*Field_grid_x[0]-Field_grid_x[1],Field_grid_x)
    Grid_location_x = np.append(Grid_location_x,2*Field_grid_x[-1]-Field_grid_x[-2])

    Nx = len(Grid_location_x)
    Nz=len(Grid_location_z)
    Nxz = (Nx-2)*(Nz-2)
    J=1
    Epsilon = 1/36/math.pi * 1e-9
    Mu = 4 * math.pi *1e-7
    Field_sigma_temp = np.reshape(Field_sigma,(Z_number,X_number),order='f')
    Back_sigma = Field_sigma_temp[-1,:]
    Sigma=np.zeros((Nz,Nx))

    tmp = np.append(Back_sigma[0],Back_sigma)
    Sigma[Grid_location_z > Field_grid_z[-1],:] = np.tile(np.append(tmp,Back_sigma[-1]),(Optimal_grid_number_z2+1,1))
    Sigma[Surface+1:Surface+Z_number+1,1:-1] = Field_sigma_temp
    Sigma[Grid_location_z < 0,:] = 1e-6
    Sigma[:,0]=Sigma[:,1]
    Sigma[:,-1]=Sigma[:,-2]
    sigmaTensor = np.tile(np.reshape(Sigma,(np.size(Sigma,0),np.size(Sigma,1),1),order='f'),(1,1,2*rxNumberMT))
    X = np.zeros((Nz,Nx))
    Z = np.zeros((Nz,Nx))
    for j in range(Nz):
        for i in range(Nx):
            X[j,i] = Grid_location_x[i]
            Z[j,i] = Grid_location_z[j]

    deltaX=np.diff(Grid_location_x[1:])
    deltaZ=np.diff(Grid_location_z[1:])
    deltaXArray=np.zeros((Nz-2,Nx-2))
    deltaZArray=np.zeros((Nz-2,Nx-2))
    for j in range(Nz-2):
        for i in range(Nx-2):
            deltaXArray[j,i] = deltaX[i]
            deltaZArray[j,i] = deltaZ[j]

    deltaXTensor=np.tile(np.reshape(deltaXArray,(np.size(deltaXArray,0),np.size(deltaXArray,1),1),order='f'),(1,1,2*rxNumberMT))
    deltaZTensor=np.tile(np.reshape(deltaZArray,(np.size(deltaZArray,0),np.size(deltaZArray,1),1)),(1,1,2*rxNumberMT))
    deltaXVector=np.reshape(deltaXArray,-1,order='f')
    deltaZVector=np.reshape(deltaZArray,-1,order='f')
    meshAreaVector=deltaXVector*deltaZVector
    meshAreaArray=deltaXArray*deltaZArray
    meshFieldAreaArray=meshAreaArray[Surface-1:Surface-1+Z_number,:]
    meshAreaArray = np.reshape(meshAreaArray,-1,order='f')
    meshFieldAreaVector=np.reshape(meshFieldAreaArray,-1,order='f')
    Domain = np.zeros((Nz,Nx),dtype='int32') # Label the solution area
    for ii in range(2,Nx):
        Domain[1:Nz-1,ii-1] = np.arange((ii-2)*(Nz-2)+1,(ii-1)*(Nz-2) + 1)    # in python Domain = in matlab

    # %================ Make the table =======================
    Table = np.zeros((Nxz,2),dtype='int32')   # Table[] in python = in matlab - 1
    q = 0
    for j in range (Nx):
        for i in range(Nz):
            if (Domain[i, j] > 0):
                Table[q, 0] = i
                Table[q, 1] = j
                q = q + 1
    HsHrjHrkFieldVec = np.zeros((gridNumber,2*rxNumberMT,freqNumberMT), dtype='complex')
    firstZLocation=Grid_location_z[0]
    Rxz_index = math.ceil((Rx_z-firstZLocation)/eachlenz)    # in python Rxz_index = in matlab - 1
    ExFieldMatrix = np.zeros((Z_number,X_number,2*rxNumberMT),dtype='complex')
    EzFieldMatrix = np.zeros((Z_number,X_number,2*rxNumberMT),dtype='complex')
    ExErjErkFieldVec = []
    EzErjErkFieldVec = []

    for gr in range(freqNumberMT):

    # %============================================================

        ja=[]
        ia=[1]
        ia_temp=[]
        total_count= 1
        index = 0
        value=[0j]
        Omega = 2*math.pi*frequencyMT[gr]

        for i in range(Nxz):
            count=0
            DeltaX_LEFT = (X[ Table[i,0], Table[i,1]]-X[ Table[i,0], Table[i,1]- 1])
            DeltaZ_UP = (Z[ Table[i,0], Table[i,1]]-Z[ Table[i,0]-1 , Table[i,1]])
            DeltaZ_DOWN = (Z[ Table[i,0]+1, Table[i,1]]-Z[ Table[i,0], Table[i,1]])
            DeltaX_RIGHT = (X[ Table[i,0], Table[i,1]+1]-X[ Table[i,0], Table[i,1]])
            DeltaX_CENTER = 1/2 * (X[ Table[i,0], Table[i,1]+ 1]-X[ Table[i,0], Table[i,1]- 1])
            DeltaZ_CENTER = 1/2 * (Z[ Table[i,0]+1, Table[i,1]]-Z[ Table[i,0]-1, Table[i,1]])

            if( Domain[ Table[i,0], Table[i,1]-1]>0 ):
                ia_temp.append(i)
                ja.append(Domain[ Table[i,0], Table[i,1]-1] - 1)
                value.append(1/DeltaX_LEFT/DeltaX_CENTER \
                /(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0], Table[i,1]-1] -1j*Omega*Epsilon))
                index = index + 1
                count=count + 1



            if( Domain[ Table[i,0]-1, Table[i,1]]>0 ):
                ia_temp.append(i)
                ja.append(Domain[ Table[i,0]-1, Table[i,1]]-1)
                value.append(1/DeltaZ_UP/DeltaZ_CENTER
                /(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0]-1, Table[i,1]] -1j*Omega*Epsilon))
                index=index + 1
                count=count + 1


            if ( Domain[ Table[i,0], Table[i,1]-1]==0 ):
                ia_temp.append(i)
                ja.append(i)
                value.append(-(1/DeltaX_RIGHT)/DeltaX_CENTER/ \
                (0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0], Table[i,1]+1] -1j*Omega*Epsilon)
                -(1/DeltaZ_DOWN/(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0]+1, Table[i,1]] -1j*Omega*Epsilon)
                  +1/DeltaZ_UP/(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0]-1, Table[i,1]] -1j*Omega*Epsilon))/DeltaZ_CENTER \
                + (1j*Omega*Mu))
                index=index+1
                count=count+1
            elif (Domain[ Table[i,0], Table[i,1]+1]==0 ):
                ia_temp.append(i)
                ja.append(i)
                value.append(-(1/DeltaX_LEFT)/DeltaX_CENTER/ \
                (0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0], Table[i,1]-1] -1j*Omega*Epsilon)
                -(1/DeltaZ_DOWN/(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0]+1, Table[i,1]] -1j*Omega*Epsilon)
                  +1/DeltaZ_UP/(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0]-1, Table[i,1]] -1j*Omega*Epsilon))/DeltaZ_CENTER \
                + (1j*Omega*Mu))
                index=index+1
                count=count+1
            else:
                ia_temp.append(i)
                ja.append(i)
                value.append(-(1/DeltaX_RIGHT/(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0], Table[i,1]+1] -1j*Omega*Epsilon)
                                 +1/DeltaX_LEFT/(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0], Table[i,1]-1] -1j*Omega*Epsilon))/DeltaX_CENTER
                -(1/DeltaZ_DOWN/(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0]+1, Table[i,1]] -1j*Omega*Epsilon)
                  +1/DeltaZ_UP/(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0]-1, Table[i,1]] -1j*Omega*Epsilon))/DeltaZ_CENTER \
                + (1j*Omega*Mu))
                index=index+1
                count=count+1


            if( Domain[ Table[i,0]+1, Table[i,1]]>0):
                ia_temp.append(i)
                ja.append(Domain[ Table[i,0]+1, Table[i,1]] - 1)
                value.append(1/DeltaZ_DOWN/DeltaZ_CENTER \
                /(0.5*Sigma[ Table[i,0]+1, Table[i,1]]+0.5*Sigma[ Table[i,0], Table[i,1]] -1j*Omega*Epsilon))
                index = index + 1
                count = count + 1


            if( Domain[ Table[i,0], Table[i,1]+1]>0):
                ia_temp.append(i)
                ja.append(Domain[ Table[i,0], Table[i,1]+1] - 1)
                value.append(1/DeltaX_RIGHT/DeltaX_CENTER \
                /(0.5*Sigma[ Table[i,0], Table[i,1]]+0.5*Sigma[ Table[i,0], Table[i,1]+1] -1j*Omega*Epsilon))
                index = index + 1
                count = count + 1

            total_count = total_count + count
            ia.append(total_count)

        ia_temp = np.reshape(np.array(ia_temp),-1)
        ja = np.reshape(np.array(ja),-1)
        value= np.reshape(np.array(value[1:]),-1)
        D = csr_matrix((value,(ia_temp, ja)), shape = (Nxz,Nxz))
        UPointJCurrent=np.zeros(((Nz-2)*X_number,rxNumberMT))
        UPointKCurrent=np.zeros(((Nz-2)*X_number,rxNumberMT),dtype='complex')   # in python rxIndexMT = in matlab - 1?
        for j in range(rxNumberMT):
            # %         UPointJCurrent(Domain(Surface+1,rxIndexMT[j]+1),j)= (-0.5)./meshAreaArray(Domain(Surface+1,rxIndexMT[j]+1))
            # %         UPointJCurrent(Domain(Surface+2,rxIndexMT[j]+1),j)= (-0.5)./meshAreaArray(Domain(Surface+1,rxIndexMT[j]+1))
            # %         UPointKCurrent(Domain(Surface+1,rxIndexMT[j]+1),j)=1/1/(deltaZVector(Domain(Surface+1,rxIndexMT[j]+1)))./(Sigma(Surface+1,rxIndexMT[j]+1)-1j*Omega*Epsilon)./meshAreaArray(Domain(Rxz_index+1,rxIndexMT[j]+1))
            # %         UPointKCurrent(Domain(Surface+2,rxIndexMT[j]+1),j)=-1/1/(deltaZVector(Domain(Surface+1,rxIndexMT[j]+1)))./(Sigma(Surface+1,rxIndexMT[j]+1)-1j*Omega*Epsilon)./meshAreaArray(Domain(Rxz_index+1,rxIndexMT[j]+1))
            UPointJCurrent[Domain[Surface+1,rxIndexMT[j]+1] -1,j]= (-1)/meshAreaArray[Domain[Surface+1,rxIndexMT[j]+1]-1]
            # %         UPointJCurrent(Domain(Surface,rxIndexMT[j]+1),j)= (-0.5)./meshAreaArray(Domain(Surface+1,rxIndexMT[j]+1))
            UPointKCurrent[Domain[Surface,rxIndexMT[j]+1] -1,j]=1/1/(deltaZVector[Domain[Surface+1,rxIndexMT[j]+1]-1])/(Sigma[Surface+1,rxIndexMT[j]+1]-1j*Omega*Epsilon)/meshAreaArray[Domain[Rxz_index+1,rxIndexMT[j]+1]-1]
            UPointKCurrent[Domain[Surface+1,rxIndexMT[j]+1] -1,j]=-1/1/(deltaZVector[Domain[Surface+1,rxIndexMT[j]+1]-1])/(Sigma[Surface+1,rxIndexMT[j]+1]-1j*Omega*Epsilon)/meshAreaArray[Domain[Rxz_index+1,rxIndexMT[j]+1]-1]


        # URight=[UPointJCurrent,UPointKCurrent]
        URight = np.zeros(((Nz-2)*X_number,2*rxNumberMT),dtype='complex')
        URight[:,:rxNumberMT] = UPointJCurrent
        URight[:,rxNumberMT:] = UPointKCurrent
        Ux = linalg.spsolve(D,URight)
        UxMatrix=np.reshape(Ux,(Nz-2,Nx-2,np.size(URight,1)), order='f')

        ExArray = 1/(0.5*sigmaTensor[1:-2,1:-1,:]+0.5*sigmaTensor[2:-1,1:-1,:]-1j*Omega*Epsilon) * (UxMatrix[1:,:,:]-UxMatrix[0:-1,:,:])/deltaZTensor[0:-1,:,:]
        EzArray = 1/(0.5*sigmaTensor[1:-1,1:-2,:]+0.5*sigmaTensor[1:-1,2:-1,:]-1j*Omega*Epsilon) * (UxMatrix[:,1:,:]-UxMatrix[:,0:-1,:])/deltaZTensor[:,0:-1,:]
        ExFieldMatrix = ExArray[Surface:Surface+Z_number,:,:]
        EzFieldMatrix[:,0:-1,:]  = EzArray[Surface:Surface+Z_number,:,:]

        HxFieldGrid=UxMatrix[Surface:Surface+Z_number,:,:]

        # ExErjErkFieldVec[:,:,gr]=np.reshape(ExFieldMatrix,(gridNumber,np.size(URight,1)),order='f')
        # EzErjErkFieldVec[:,:,gr]=np.reshape(EzFieldMatrix,(gridNumber,np.size(URight,1)),order='f')
    # % 在这里算电场
        ExErjErkFieldVec.append(np.reshape(ExFieldMatrix,(gridNumber,np.size(URight,1)),order='f'))
        EzErjErkFieldVec.append(np.reshape(EzFieldMatrix,(gridNumber,np.size(URight,1)),order='f'))

    ExErjErkFieldVec = np.array(ExErjErkFieldVec).transpose((1,2,0))
    EzErjErkFieldVec = np.array(EzErjErkFieldVec).transpose((1,2,0))
    partialEPartialSigmaArray=np.zeros((gridNumber,rxNumberMT,freqNumberMT),dtype='complex')
    partialHPartialSigmaArray=np.zeros((gridNumber,rxNumberMT,freqNumberMT),dtype='complex')
    ExFieldVectorReshape=np.reshape(ExFieldVector,(gridNumber,1,freqNumberMT),order='f')
    EzFieldVectorReshape=np.reshape(EzFieldVector,(gridNumber,1,freqNumberMT),order='f')
    tempMeshFieldAreaVector = np.tile(np.reshape(meshFieldAreaVector,(len(meshFieldAreaVector),1,1)),(1,1,freqNumberMT))
    for i in range(rxNumberMT):
        B=ExErjErkFieldVec[:,i,:]   # B is two dimensional
        partialHPartialSigmaArray[:,i,:]=\
        (ExFieldVectorReshape.squeeze()*ExErjErkFieldVec[:,i,:]+EzFieldVectorReshape.squeeze()*EzErjErkFieldVec[:,i,:])*tempMeshFieldAreaVector.squeeze()
        partialEPartialSigmaArray[:,i,:]=\
        (ExFieldVectorReshape.squeeze()*ExErjErkFieldVec[:,len(rxIndexMT)+i,:]+EzFieldVectorReshape.squeeze()*EzErjErkFieldVec[:,len(rxIndexMT)+i,:])*tempMeshFieldAreaVector.squeeze()

    partialEPartialSigma=np.reshape(partialEPartialSigmaArray,(gridNumber,dataLengthMT),order='f')
    partialHPartialSigma=np.reshape(partialHPartialSigmaArray,(gridNumber,dataLengthMT),order='f')
    # % appResistivityRepmat=repmat(10.^(Logrho_f),gridNumber,1)
    # % appResistivityRepmat=repmat((Logrho_f),gridNumber,1)

    EobsVectorRepmat=np.tile(EobsVector,(gridNumber,1))
    HobsVectorRepmat=np.tile(HobsVector,(gridNumber,1))
    partialAppResistivityPartialSigma=2*(np.real(partialEPartialSigma/EobsVectorRepmat)
     -np.real(partialHPartialSigma/HobsVectorRepmat))
    FieldRhoRepmat=np.tile(10**np.reshape(LogField_rho,(-1,1)),(1,dataLengthMT))
    # % FieldRhoRepmat=np.tile(LogField_rho,1,dataLengthMT)
    # % jacobianRho=(-partialAppResistivityPartialSigma./FieldRhoRepmat./appResistivityRepmat).'
    jacobianRho=(-partialAppResistivityPartialSigma/FieldRhoRepmat).T
    jacobianPhi = (math.log(10)/FieldRhoRepmat * (np.imag(partialEPartialSigma/EobsVectorRepmat)
                                               -np.imag(partialHPartialSigma/HobsVectorRepmat))).T

    return jacobianRho,jacobianPhi

def ComputeJacobianFunc_z_tf(EobsVector, HobsVector, EFieldVectorf, frequencyMT, LogField_rho, Rx,
                          Field_grid_x, Field_grid_z, X_number, Z_number, rxIndexMT,ia_temp, ja, value,Domain, index1):
    freqNumberMT = len(frequencyMT)
    rxNumberMT = len(rxIndexMT)
    Field_sigma = 1 / 10 ** LogField_rho
    gridNumber = X_number * Z_number
    dataLengthMT = rxNumberMT* freqNumberMT
    Rx_z = Rx[0]
    Rx_x = Rx[1]
    Grid_air_num = 7
    eachlenz = Field_grid_z[1] - Field_grid_z[0]
    Grid_z1 = eachlenz * np.arange(-1 - Grid_air_num, 0)
    Optimal_grid_number_z2 = 70
    temp2 = np.logspace(math.log10(Field_grid_z[-1] - Field_grid_z[-2]) + 0.1, 5, Optimal_grid_number_z2)
    Optimal_grid_z2 = np.cumsum(np.append(temp2, temp2[-1])) + Field_grid_z[-1]
    Grid_location_z = np.append(Grid_z1 + Field_grid_z[0], Field_grid_z)
    Grid_location_z = np.append(Grid_location_z, Optimal_grid_z2)
    Air = Grid_location_z[Grid_location_z < 0]
    Surface = np.argmax(Air)
    Grid_location_x = np.append(2 * Field_grid_x[0] - Field_grid_x[1], Field_grid_x)
    Grid_location_x = np.append(Grid_location_x, 2 * Field_grid_x[-1] - Field_grid_x[-2])
    Nx = len(Grid_location_x)
    Nz = len(Grid_location_z)
    Nxz = (Nx - 2) * (Nz - 2)
    J = 1
    Epsilon = 1 / 36 / math.pi * 1e-9
    Mu = 4 * math.pi * 1e-7
    Field_sigma_temp = tf.transpose(tf.reshape(Field_sigma, (X_number, Z_number)))
    Back_sigma = Field_sigma_temp[-1:, :]

    Sigma_concat11 = Field_sigma_temp[:Surface + 1, :] * 0 + 1e-6
    Sigma_concat2 = Field_sigma_temp
    Sigma_concat3 = Back_sigma
    for gg in range(Optimal_grid_number_z2):
        Sigma_concat3 = tf.concat((Sigma_concat3, Back_sigma), axis=0)
    Sigma = tf.concat([Sigma_concat11, Sigma_concat2, Sigma_concat3], axis=0)
    Sigma_active = Sigma[1:-1, 0]
    for pp in range(1, Sigma.shape[1]):
        Sigma_active = tf.concat((Sigma_active, Sigma[1:-1, pp]), axis=0)

    # Sigma = np.zeros((Nz, Nx))
    # Sigma[Grid_location_z > Field_grid_z[-1], :] = np.tile(
    #     np.append(np.append(Back_sigma[0], Back_sigma), Back_sigma[-1]),
    #     (Optimal_grid_number_z2 + 1, 1))
    # Sigma[Surface + 1: Surface + 1 + Z_number, 1:-1] = Field_sigma_temp
    # Sigma[Grid_location_z < 0, :] = 1e-6
    deltaX = np.diff(Grid_location_x[1:])
    deltaZ = np.diff(Grid_location_z[1:])
    deltaXArray = np.zeros((Nz - 2, Nx - 2))
    deltaZArray = np.zeros((Nz - 2, Nx - 2))
    for j in range(Nz-2):
        for i in range(Nx-2):
            deltaXArray[j, i] = deltaX[i]
            deltaZArray[j, i] = deltaZ[j]

    deltaXVector = np.reshape(deltaXArray, -1,order='f')
    deltaZVector = np.reshape(deltaZArray, -1,order='f')
    meshAreaVector = deltaXVector * deltaZVector
    meshAreaArray = deltaXArray * deltaZArray

    meshFieldAreaArray = meshAreaArray[Surface - 1:Surface - 1 + Z_number,:]
    meshFieldAreaVector = np.reshape(meshFieldAreaArray, -1,order='f')
    meshAreaArray = np.reshape(meshAreaArray, -1, order='f')
    EsErjErkFieldVec = tf.zeros((gridNumber, 2 * rxNumberMT, 1), dtype=tf.complex64)
    # EsErjErkFieldVec = np.zeros((gridNumber, 2 * rxNumberMT, freqNumberMT), dtype='complex')
    firstZLocation = Grid_location_z[0]
    Rxz_index = math.ceil((Rx_z - firstZLocation) / eachlenz)

    # Sigma_active = np.reshape(Sigma[1:- 1, 1: - 1], -1,order='f')
    for gr in range(freqNumberMT):
        print("Jacobian, no={}".format(gr))
        Omega = 2 * math.pi * frequencyMT[gr]

        value1 = value.copy()
        value1 = tf.constant(value1)
        temp1 = 0j * value1[:index1]
        temp2 = 0j * value1[index1:] - 1j * K.eval(Omega) * Mu * tf.cast(Sigma_active, dtype=tf.complex128) + K.eval(
            Omega) ** 2 * Mu * Epsilon
        temp = tf.concat((temp1, temp2), axis=0)
        value1 = value1 + temp

        # value[index1:] = value[index1:] + -1j * Omega * Mu * Sigma_active
        # value[index1:] = value[index1:] + Omega ** 2 * Mu * Epsilon
        # D = csr_matrix((value, (ia_temp, ja)))

        indices = []
        for pp in range(len(ia_temp)):
            indices.append([ia_temp[pp], ja[pp]])
        D_tf_real = tf.sparse.SparseTensor(indices, tf.math.real(value1), [len(Sigma_active), len(Sigma_active)])
        D_tf_imag = tf.sparse.SparseTensor(indices, tf.math.imag(value1), [len(Sigma_active), len(Sigma_active)])
        D_tf_real = tf.sparse.reorder(D_tf_real)
        D_tf_r = tf.sparse.to_dense(D_tf_real)
        D_tf_imag = tf.sparse.reorder(D_tf_imag)
        D_tf_i = tf.sparse.to_dense(D_tf_imag)
        D_tf = tf.cast(D_tf_r, dtype=tf.complex64) + tf.complex(0.0, 1.0) * tf.cast(D_tf_i, dtype=tf.complex64)

        # value[index1:] = value[index1:] - -1j * Omega * Mu * Sigma_active
        # value[index1:] = value[index1:] - Omega ** 2 * Mu * Epsilon

        UPointJCurrent = np.zeros(((Nz - 2) * X_number, rxNumberMT), dtype='complex')
        UPointKCurrent = np.zeros(((Nz - 2) * X_number, rxNumberMT), dtype='complex')
        for j in range(rxNumberMT):
            UPointJCurrent[int(Domain[Rxz_index, int(rxIndexMT[j])]-1), j] = -1j * Omega * Mu * (-1) / \
                                                                             meshAreaArray[int(Domain[Rxz_index, int(rxIndexMT[j])]-1)]
            UPointKCurrent[int(Domain[Rxz_index - 1, int(rxIndexMT[j])]-1), j] = 1 / 2 / \
                                                                                 (deltaZVector[int(Domain[Rxz_index, int(rxIndexMT[j])]-1)]) / meshAreaArray[int(Domain[Rxz_index, int(rxIndexMT[j])]-1)]
            # % 除以2是因为用了中心差分
            UPointKCurrent[int(Domain[Rxz_index + 1, int(rxIndexMT[j])]-1), j] = -1 / 2 / \
                                                                                 (deltaZVector[int(Domain[Rxz_index + 1, int(rxIndexMT[j])]-1)]) / meshAreaArray[int(Domain[Rxz_index, int(rxIndexMT[j])]-1)]

        URight = np.concatenate((UPointJCurrent, UPointKCurrent),axis=1)
        URight = tf.constant(URight)

        # Ux = linalg.spsolve(D, URight)
        Ux = tf.linalg.solve(D_tf, tf.cast(URight, dtype=tf.complex64))

        # UxMatrix = np.reshape(Ux, (Nz - 2, Nx - 2, np.size(URight, 1)), order='f')
        UxMatrix = tf.reshape(Ux, (tf.shape(URight)[1], Nx - 2, Nz - 2))
        UxMatrix = tf.transpose(UxMatrix, (2,1,0))

        ExFieldGrid = UxMatrix[Surface - 1:Surface - 1 + Z_number,:,:]

        ExFieldGrid = tf.transpose(ExFieldGrid, (1,0,2))
        ttmp = tf.reshape(ExFieldGrid, (gridNumber, np.size(URight, 1)))
        ttmp = tf.expand_dims(ttmp, axis=2)
        EsErjErkFieldVec = tf.concat((EsErjErkFieldVec, ttmp), axis=2)

        # EsErjErkFieldVec[:,:, gr]=np.reshape(ExFieldGrid, (gridNumber, np.size(URight, 1)),order='f')
    EsErjErkFieldVec = EsErjErkFieldVec[:, :, 1:]
    # partialEPartialSigmaArray = np.zeros((gridNumber, rxNumberMT, freqNumberMT),dtype='complex')
    # partialHPartialSigmaArray = np.zeros((gridNumber, rxNumberMT, freqNumberMT),dtype='complex')
    EFieldVectorReshape = tf.transpose(tf.reshape(EFieldVectorf, (freqNumberMT, gridNumber)))
    EFieldVectorReshape = tf.expand_dims(EFieldVectorReshape, axis=1)

    print("至此")

    # tempMeshFieldAreaVector = np.tile(np.reshape(meshFieldAreaVector,(-1,1,1),order='f'), (1, 1, freqNumberMT))
    meshFieldAreaVector = tf.constant(meshFieldAreaVector, dtype=tf.complex64)
    tempMeshFieldAreaVector = tf.tile(tf.reshape(meshFieldAreaVector,(-1,1)), tf.constant([1, freqNumberMT], tf.int32))
    tempMeshFieldAreaVector = tf.expand_dims(tempMeshFieldAreaVector, axis=1)

    partialEPartialSigmaArray = EFieldVectorReshape * tf.expand_dims(EsErjErkFieldVec[:, 0, :], axis=1)\
                                *tempMeshFieldAreaVector
    partialHPartialSigmaArray = EFieldVectorReshape \
                  * tf.expand_dims(EsErjErkFieldVec[:, len(rxIndexMT) + 0, :], axis=1) * tempMeshFieldAreaVector
    for i in range(1, rxNumberMT):
        tmp = EFieldVectorReshape * tf.expand_dims(EsErjErkFieldVec[:, i, :], axis=1)\
                                *tempMeshFieldAreaVector
        partialEPartialSigmaArray = tf.concat((partialEPartialSigmaArray, tmp), axis=1)
        # partialEPartialSigmaArray[:, i,:] = EFieldVectorReshape* EsErjErkFieldVec[:, i,:]
        # partialEPartialSigmaArray[:, i,:] = partialEPartialSigmaArray[:, i,:]*tempMeshFieldAreaVector
        tmp = EFieldVectorReshape * tf.expand_dims(EsErjErkFieldVec[:, len(rxIndexMT) + i,:], axis=1)\
              *tempMeshFieldAreaVector
        partialHPartialSigmaArray = tf.concat((partialHPartialSigmaArray, tmp), axis=1)
        # partialHPartialSigmaArray[:, i,:] = \
        #     EFieldVectorReshape * EsErjErkFieldVec[:, len(rxIndexMT) + i,:]*tempMeshFieldAreaVector

    # partialEPartialSigma = np.reshape(partialEPartialSigmaArray, (gridNumber, dataLengthMT),order='f')
    # partialHPartialSigma = np.reshape(partialHPartialSigmaArray, (gridNumber, dataLengthMT), order='f')
    partialEPartialSigma = tf.transpose(partialEPartialSigmaArray, (0,2,1))
    partialEPartialSigma = tf.reshape(partialEPartialSigma, (gridNumber, dataLengthMT))
    partialHPartialSigma = tf.transpose(partialHPartialSigmaArray, (0,2,1))
    partialHPartialSigma = tf.reshape(partialHPartialSigma, (gridNumber, dataLengthMT))

    # EobsVectorRepmat = np.tile(EobsVector, (gridNumber, 1))
    # HobsVectorRepmat = np.tile(HobsVector, (gridNumber, 1))
    # tmp1 = tf.constant([[1,2,3],[4,5,6]], tf.int32)
    EobsVector = tf.transpose(EobsVector)
    HobsVector = tf.transpose(HobsVector)
    tmp2 = tf.constant([gridNumber, 1], tf.int32)
    EobsVectorRepmat = tf.tile(EobsVector, tmp2)
    HobsVectorRepmat = tf.tile(HobsVector, tmp2)
    HobsVectorRepmat = tf.cast(HobsVectorRepmat, dtype=tf.complex64)

    partialAppResistivityPartialSigma = 2 * (tf.math.real(partialEPartialSigma / EobsVectorRepmat) -
                                             tf.math.real(partialHPartialSigma / HobsVectorRepmat))
    # FieldRhoRepmat = np.tile(10** np.reshape(LogField_rho,(-1,1),order='f'), (1, dataLengthMT))
    FieldRhoRepmat = tf.tile(10 ** tf.reshape(LogField_rho, (-1, 1)), tf.constant([1, dataLengthMT], tf.int32))

    jacobianRho = -partialAppResistivityPartialSigma / FieldRhoRepmat
    jacobianPhi = -(tf.math.log(tf.cast(10, dtype=tf.float32)) / FieldRhoRepmat * (tf.math.imag(partialEPartialSigma / EobsVectorRepmat)
                                                   - tf.math.imag(partialHPartialSigma / HobsVectorRepmat)))
    return tf.transpose(jacobianRho), tf.transpose(jacobianPhi)

import math
import numpy as np
import time
from scipy.ndimage import filters
import random
# from pathos.multiprocessing import ProcessingPoll as Pool
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import multiprocessing

# def MT2DFWD2_wrapper(gr, freq,Field_rho,Rx,Field_grid_x, Field_grid_z, X_number,Z_number,Rx_index):
#     EFieldVector = np.zeros((X_number*Z_number,len(frequencyMT)))
#     Eobs = np.zeros((len(rxIndexMT),len(frequencyMT)))
#     Hobs = np.zeros((len(rxIndexMT),len(frequencyMT)))
#     rho_f_mat = np.zeros((len(rxIndexMT),len(frequencyMT)))
#     phase_f_mat = np.zeros((len(rxIndexMT),len(frequencyMT)))
#     [EfieldVector_o, Eobs_o, Hobs_o, rho_f_mat_o, phase_f_mat_o] = MT2DFWD2(gr, freq, Field_rho, Rx, Field_grid_x,
#         Field_grid_z, X_number, Z_number, Rx_index)
#     EFieldVector[:,gr] = EFieldVector_o
#     Eobs[:,gr] = Eobs_o
#     Hobs[:,gr] = Hobs_o
#     rho_f_mat[:,gr] = rho_f_mat_o
#     phase_f_mat[:,gr] = phase_f_mat_o
#     rho_f = np.log10(reshape(rho_f_mat, -1))
#     phase_f = np.reshape(phase_f_mat, -1)
#     data_f = np.append(rho_f, phase_f)
#     # % rho_f = (reshape(rho_f_mat, Rx_num * freq_num, 1).
#     # ') \
#     #                                                 % rho_f=rho_f_mat
#     EFieldVectorf = EFieldVector
#     EobsVector = np.reshape(Eobs, -1)
#     HobsVector = np.reshape(Hobs, -1)
#
#     return data_f,EobsVector,HobsVector,EFieldVectorf

def MT2SparseEquationSetUp_zhhy(Field_grid_x, Field_grid_z):
    Grid_air_num = 7
    eachlenz = Field_grid_z[1] - Field_grid_z[0]
    Grid_z1 = eachlenz * np.arange(-1 - Grid_air_num, 0)
    Optimal_grid_number_z2 = 70
    temp2 = np.logspace(np.log10(Field_grid_z[-1] - Field_grid_z[-2]) + 0.1, 5, Optimal_grid_number_z2)
    Optimal_grid_z2 = np.cumsum(np.append(temp2, temp2[-1])) + Field_grid_z[-1]
    Grid_location_z = np.append(Grid_z1 + Field_grid_z[0], Field_grid_z)
    Grid_location_z = np.append(Grid_location_z, Optimal_grid_z2)
    Grid_location_x = np.append(2 * Field_grid_x[0] - Field_grid_x[1], Field_grid_x)
    Grid_location_x = np.append(Grid_location_x, 2 * Field_grid_x[-1] - Field_grid_x[-2])

    Nx = len(Grid_location_x)
    Nz = len(Grid_location_z)
    Nxz = (Nx - 2) * (Nz - 2)
    X = np.zeros((Nz, Nx))
    Z = np.zeros((Nz, Nx))
    for j in range(Nz):
        for i in range(Nx):
            X[j, i] = Grid_location_x[i]
            Z[j, i] = Grid_location_z[j]

    Area = np.zeros((Nz, Nx))  # Label the solution area  first label = 1
    for ii in range(1, Nx - 1):
        Area[1: Nz - 1, ii] = np.arange((ii - 1) * (Nz - 2) + 1, (ii) * (Nz - 2) + 1)
    Area = np.array(Area, dtype='int32')
    # = == == == == == == == = Make the table == == == == == == == == == == == =
    Table = np.zeros((Nxz, 2))
    q = 0
    for j in range(Nx):
        for i in range(Nz):
            if (Area[i, j] > 0):
                Table[q, 0] = i
                Table[q, 1] = j
                q = q + 1


    # parfor
    # gr = 1:length(freq)
    ub = [[0 + 0j] * Nxz]
    Ub = np.reshape(np.array(ub), -1)
    # = == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    ja = []
    ia = [1]
    ia_temp = []
    total_count = 1
    index = 0
    value = [0j]

    MT2ts1 = time.time()
    for i in range(Nxz):
        DeltaX_LEFT = X[int(Table[i, 0]), int(Table[i, 1])] - X[int(Table[i, 0]), int(Table[i, 1]) - 1]
        DeltaZ_UP = (Z[int(Table[i, 0]), int(Table[i, 1])] - Z[int(Table[i, 0]) - 1, int(Table[i, 1])])
        DeltaZ_DOWN = (Z[int(Table[i, 0]) + 1, int(Table[i, 1])] - Z[int(Table[i, 0]), int(Table[i, 1])])
        DeltaX_RIGHT = (X[int(Table[i, 0]), int(Table[i, 1]) + 1] - X[int(Table[i, 0]), int(Table[i, 1])])
        DeltaX_CENTER = 1 / 2 * (X[int(Table[i, 0]), int(Table[i, 1]) + 1] - X[int(Table[i, 0]), int(Table[i, 1]) - 1])
        DeltaZ_CENTER = 1 / 2 * (Z[int(Table[i, 0]) + 1, int(Table[i, 1])] - Z[int(Table[i, 0]) - 1, int(Table[i, 1])])

        if (Area[int(Table[i, 0]), int(Table[i, 1]) - 1] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]), int(Table[i, 1]) - 1] - 1)
            value.append(1 / DeltaX_LEFT / DeltaX_CENTER)
            index = index + 1

        if (Area[int(Table[i, 0]) - 1, int(Table[i, 1])] == 0):
            Ub[i] = 1

        if (Area[int(Table[i, 0]) - 1, int(Table[i, 1])] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]) - 1, int(Table[i, 1])] - 1)
            value.append(1 / DeltaZ_UP / DeltaZ_CENTER)
            index = index + 1

        if (Area[int(Table[i, 0]) + 1, int(Table[i, 1])] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]) + 1, int(Table[i, 1])] - 1)
            value.append(1 / DeltaZ_DOWN / DeltaZ_CENTER)
            index = index + 1

        if (Area[int(Table[i, 0]), int(Table[i, 1]) + 1] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]), int(Table[i, 1]) + 1] - 1)
            value.append(1 / DeltaX_RIGHT / DeltaX_CENTER)
            index = index + 1

    index1 = index

    for i in range(Nxz):
        DeltaX_LEFT = X[int(Table[i, 0]), int(Table[i, 1])] - X[int(Table[i, 0]), int(Table[i, 1]) - 1]
        DeltaZ_UP = (Z[int(Table[i, 0]), int(Table[i, 1])] - Z[int(Table[i, 0]) - 1, int(Table[i, 1])])
        DeltaZ_DOWN = (Z[int(Table[i, 0]) + 1, int(Table[i, 1])] - Z[int(Table[i, 0]), int(Table[i, 1])])
        DeltaX_RIGHT = (X[int(Table[i, 0]), int(Table[i, 1]) + 1] - X[int(Table[i, 0]), int(Table[i, 1])])
        DeltaX_CENTER = 1 / 2 * (X[int(Table[i, 0]), int(Table[i, 1]) + 1] - X[int(Table[i, 0]), int(Table[i, 1]) - 1])
        DeltaZ_CENTER = 1 / 2 * (Z[int(Table[i, 0]) + 1, int(Table[i, 1])] - Z[int(Table[i, 0]) - 1, int(Table[i, 1])])
        if (Area[int(Table[i, 0]), int(Table[i, 1]) - 1] == 0):
            ia_temp.append(i)
            ja.append(i)
            value.append(-(1 / DeltaX_RIGHT) / DeltaX_CENTER - \
                         (1 / DeltaZ_DOWN + 1 / DeltaZ_UP) / DeltaZ_CENTER)
            index = index + 1
        elif (Area[int(Table[i, 0]), int(Table[i, 1]) + 1] == 0):
            ia_temp.append(i)
            ja.append(i)
            value.append(-(1 / DeltaX_LEFT) / DeltaX_CENTER \
                         - (1 / DeltaZ_DOWN + 1 / DeltaZ_UP) / DeltaZ_CENTER)
            index = index + 1
        else:
            ia_temp.append(i)
            ja.append(i)
            value.append(-(1 / DeltaX_RIGHT + 1 / DeltaX_LEFT) / DeltaX_CENTER \
                         - (1 / DeltaZ_DOWN + 1 / DeltaZ_UP) / DeltaZ_CENTER)
            index = index + 1
    MT2ts2 = time.time()
    # print("No.{} iter".format(gr))
    print("MT2FWD equation set-up time:{} s".format(MT2ts2 - MT2ts1))
    ia_temp = np.reshape(np.array(ia_temp), -1)
    ja = np.reshape(np.array(ja), -1)
    value = np.reshape(np.array(value[1:], dtype = complex), -1)
    return ia_temp, ja, value, Ub, Area, index1, Z

def MT2DFWD2_zhhy(MT2DFWD2dic, gr):
    freq = MT2DFWD2dic['freq']
    Field_rho=MT2DFWD2dic['Field_rho']
    Rx = MT2DFWD2dic['Rx']
    Field_grid_x = MT2DFWD2dic['Field_grid_x']
    Field_grid_z = MT2DFWD2dic['Field_grid_z']
    X_number = MT2DFWD2dic['X_number']
    Z_number = MT2DFWD2dic['Z_number']
    Rx_index = MT2DFWD2dic['Rx_index']
    ia_temp = MT2DFWD2dic['ia_temp']
    ja = MT2DFWD2dic['ja']
    value = MT2DFWD2dic['value']
    Ub = MT2DFWD2dic['Ub']
    Area = MT2DFWD2dic['Area']
    index1 = MT2DFWD2dic['index1']
    Z = MT2DFWD2dic['Z']
    Rxx_index = Rx_index
    Field_sigma = 1 / 10 ** Field_rho
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
    Field_sigma_temp = np.reshape(Field_sigma, (Z_number, X_number),order='f')
    Back_sigma = Field_sigma_temp[-1,:]
    Sigma = np.zeros((Nz, Nx))
    Sigma[Grid_location_z > Field_grid_z[-1], :] = np.tile(
        np.append(np.append(Back_sigma[0], Back_sigma), Back_sigma[-1]),
        (Optimal_grid_number_z2 + 1, 1))
    Sigma[Surface + 1: Surface + 1 + Z_number, 1:-1] = Field_sigma_temp
    Sigma[Grid_location_z < 0, :] = 1e-6
    Sigma_active = np.reshape(Sigma[1:- 1, 1: - 1], -1,order='f')
    MT2ts2 = time.time()
    Omega = 2 * math.pi * freq[gr]
    Ub1 = np.zeros(len(Ub), dtype='complex')
    Ub1[Ub == 1] = 1j * Omega * Mu * J
    value1 = value.copy()
    value1[index1:] = value1[index1:] + -1j * Omega * Mu * Sigma_active
    value1[index1:] = value1[index1:] + Omega ** 2 * Mu * Epsilon
    D = csr_matrix((value1, (ia_temp, ja)),shape=(len(Sigma_active),len(Sigma_active)))
    Ux = linalg.spsolve(D, Ub1)
    MT2ts3 = time.time()
    # print("MT2FWD equation solving time:{} s".format(MT2ts3 - MT2ts2))
    # Ux = D\Ub
    Rxz_index = math.ceil((Rx_z - Grid_location_z[0]) / eachlenz)
    EDomainArray = np.reshape(Ux, (Nz - 2, Nx - 2), order='f')
    EFieldArray = EDomainArray[Surface:Surface + Z_number, :]
    EFieldVector_in = np.reshape(EFieldArray, -1,order='f')
    Rxx_index_list = np.array(Rxx_index.tolist(), dtype='int32').tolist()
    Eobs_in = Ux[(Area[Surface + 1, Rxx_index_list] - 1).tolist()]   # 2023年8月21日这里应该是(Area[Surface，不是(Area[Surface+1,...？
    Delta_z = Z[Surface + 1, Rxx_index_list] - Z[Surface, Rxx_index_list]
    Hobs_in = -1 / 1j / Omega / Mu * (
            Ux[Area[Surface + 2, Rxx_index_list] - 1] - Ux[Area[Surface, Rxx_index_list] - 1]) / 2 / Delta_z
    # % Rxx_index_center = round(Rx_x / eachlenx) + 1
    # % EdivH = Eobs(Rxx_index). / Hobs(Rxx_index)
    EdivH = Eobs_in / Hobs_in
    rho_f_mat_in = np.log10(1 / Omega / Mu * (abs(EdivH) ** 2))
    phase_f_mat_in = np.zeros(len(EdivH))
    for i in range(len(EdivH)):
        phase_f_mat_in[i] = math.atan(EdivH[i].imag / EdivH[i].real)
    data_f = np.append(rho_f_mat_in, phase_f_mat_in)
    return {'data_f': data_f, 'EFieldVector_in': EFieldVector_in, 'Eobs_in': Eobs_in, 'Hobs_in': Hobs_in}

def MT2DFWD2(MT2DFWD2dic, gr):
    freq = MT2DFWD2dic['freq']
    Field_rho=MT2DFWD2dic['Field_rho']
    Rx = MT2DFWD2dic['Rx']
    Field_grid_x = MT2DFWD2dic['Field_grid_x']
    Field_grid_z = MT2DFWD2dic['Field_grid_z']
    X_number = MT2DFWD2dic['X_number']
    Z_number = MT2DFWD2dic['Z_number']
    Rx_index = MT2DFWD2dic['Rx_index']
    Field_sigma = 1 / 10 ** Field_rho
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
    Field_sigma_temp = np.reshape(Field_sigma, (Z_number, X_number),order='f')
    Back_sigma = Field_sigma_temp[-1,:]
    Sigma = np.zeros((Nz, Nx))
    Sigma[Grid_location_z > Field_grid_z[-1],:] = np.tile(np.append(np.append(Back_sigma[0], Back_sigma), Back_sigma[-1]),
                                                            (Optimal_grid_number_z2 + 1, 1))

# % Sigma(Grid_location_z > Field_grid_z(end),:)=Back_sigma
# % Sigma(:, Grid_location_x < Field_grid_x(1)) = Back_sigma
# % Sigma(:, Grid_location_x > Field_grid_x(end)) = Back_sigma
    Sigma[Surface + 1 : Surface + 1 + Z_number , 1:-1 ] = Field_sigma_temp
    Sigma[Grid_location_z < 0,:] = 1e-6
    X = np.zeros((Nz,Nx))
    Z = np.zeros((Nz,Nx))
    for j in range(Nz):
        for i in range(Nx):
            X[j,i] = Grid_location_x[i]
            Z[j,i] = Grid_location_z[j]

    Area = np.zeros((Nz, Nx))  # Label the solution area  first label = 1
    for ii in range(1,Nx - 1):
        Area[1: Nz - 1, ii] = np.arange((ii - 1) * (Nz - 2) + 1, (ii) * (Nz - 2)+1)
    Area = np.array(Area, dtype='int32')
# = == == == == == == == = Make the table == == == == == == == == == == == =
    Table = np.zeros((Nxz, 2))
    q = 0
    for j in range (Nx):
        for i in range(Nz):
            if (Area[i, j] > 0):
                Table[q, 0] = i
                Table[q, 1] = j
                q = q + 1

    Rxx_index = Rx_index
# parfor
# gr = 1:length(freq)
    ub = [[0+0j]*Nxz]
    Ub = np.reshape(np.array(ub),-1)
# = == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    ja = []
    ia = [1]
    ia_temp = []
    total_count = 1
    index = 0
    value = [0j]
    Omega = 2 * math.pi * freq[gr]
    MT2ts1 = time.time()
    for i in range(Nxz):
        count = 0
        DeltaX_LEFT = X[int(Table[i, 0]), int(Table[i,1])] - X[int(Table[i,0]), int(Table[i,1]) - 1]
        DeltaZ_UP = (Z[int(Table[i,0]), int(Table[i,1])] - Z[int(Table[i,0]) - 1, int(Table[i,1])])
        DeltaZ_DOWN = (Z[int(Table[i,0]) + 1, int(Table[i,1])] - Z[int(Table[i,0]), int(Table[i,1])])
        DeltaX_RIGHT = (X[int(Table[i,0]), int(Table[i,1]) + 1] - X[int(Table[i,0]), int(Table[i,1])])
        DeltaX_CENTER = 1 / 2 * (X[int(Table[i,0]), int(Table[i,1]) + 1] - X[int(Table[i,0]), int(Table[i,1]) - 1])
        DeltaZ_CENTER = 1 / 2 * (Z[int(Table[i,0]) + 1, int(Table[i,1])] - Z[int(Table[i,0]) - 1, int(Table[i,1])])
    
        if (Area[int(Table[i, 0]), int(Table[i, 1]) - 1] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]), int(Table[i, 1]) - 1]-1)
            value.append(1 / DeltaX_LEFT / DeltaX_CENTER)
            index = index + 1
            count = count + 1
    
        if (Area[int(Table[i, 0]) - 1, int(Table[i, 1])] == 0):
            Ub[i] = 1j * Omega * Mu * J

    
        if (Area[int(Table[i, 0]) - 1, int(Table[i, 1])] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]) - 1, int(Table[i, 1])]-1)
            value.append(1 / DeltaZ_UP / DeltaZ_CENTER)
            index = index + 1
            count = count + 1
    
        if (Area[int(Table[i, 0]), int(Table[i, 1]) - 1] == 0):
            ia_temp.append(i)
            ja.append(i)
            value.append(-(1 / DeltaX_RIGHT) / DeltaX_CENTER - \
            (1 / DeltaZ_DOWN + 1 / DeltaZ_UP) / DeltaZ_CENTER + \
            (-1j * Omega * Mu * Sigma[int(Table[i, 0]), int(Table[i, 1])]+Omega**2 * Mu * Epsilon))
            index = index + 1
            count = count + 1
        elif(Area[int(Table[i, 0]), int(Table[i, 1]) + 1] == 0):
            ia_temp.append(i)
            ja.append(i)
            value.append(-(1 / DeltaX_LEFT) / DeltaX_CENTER \
            -(1 / DeltaZ_DOWN + 1 / DeltaZ_UP) / DeltaZ_CENTER \
            + (-1j * Omega * Mu * Sigma[int(Table[i, 0]), int(Table[i, 1])]+Omega ** 2 * Mu * Epsilon))
            index = index + 1
            count = count + 1
        else:
            ia_temp.append(i)
            ja.append(i)
            value.append(-(1 / DeltaX_RIGHT + 1 / DeltaX_LEFT) / DeltaX_CENTER \
            -(1 / DeltaZ_DOWN + 1 / DeltaZ_UP) / DeltaZ_CENTER \
            + (-1j * Omega * Mu * Sigma[int(Table[i, 0]), int(Table[i, 1])]+Omega ** 2 * Mu * Epsilon))
            index = index + 1
            count = count + 1

    
        if (Area[int(Table[i, 0]) + 1, int(Table[i, 1])] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]) + 1, int(Table[i, 1])]-1)
            value.append(1 / DeltaZ_DOWN / DeltaZ_CENTER)
            index = index + 1
            count = count + 1
    
        if (Area[int(Table[i, 0]), int(Table[i, 1]) + 1] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]), int(Table[i, 1]) + 1]-1)
            value.append(1 / DeltaX_RIGHT / DeltaX_CENTER)
            index = index + 1
            count = count + 1

        total_count = total_count + count
        ia.append(total_count)
    MT2ts2 = time.time()
    print(multiprocessing.current_process().name)
    # print("No.{} iter".format(gr))
    print("MT2FWD equation set-up time:{} s".format(MT2ts2-MT2ts1))
    ia_temp = np.reshape(np.array(ia_temp),-1)
    ja = np.reshape(np.array(ja),-1)
    value= np.reshape(np.array(value[1:]),-1)
    D = csr_matrix((value,(ia_temp, ja)), shape = (Nxz,Nxz))
    Ux = linalg.spsolve(D,Ub)
    MT2ts3 = time.time()
    print("MT2FWD equation solving time:{} s".format(MT2ts3-MT2ts2))
    # Ux = D\Ub
    Rxz_index = math.ceil((Rx_z - Grid_location_z[0]) / eachlenz)
    EDomainArray = np.reshape(Ux, (Nz - 2, Nx - 2),order='f')
    EFieldArray = EDomainArray[Surface:Surface + Z_number,:]
    EFieldVector_in = np.reshape(EFieldArray, -1)
    Rxx_index_list = np.array(Rxx_index.tolist(),dtype='int32').tolist()
    Eobs_in =Ux[(Area[Surface + 1, Rxx_index_list]-1).tolist()]
    Delta_z = Z[Surface + 1, Rxx_index_list] - Z[Surface, Rxx_index_list]
    Hobs_in  = -1 / 1j / Omega / Mu * (Ux[Area[Surface + 2, Rxx_index_list]-1] - Ux[Area[Surface, Rxx_index_list]-1]) / 2 / Delta_z
    # % Rxx_index_center = round(Rx_x / eachlenx) + 1
    # % EdivH = Eobs(Rxx_index). / Hobs(Rxx_index)
    EdivH = Eobs_in / Hobs_in
    rho_f_mat_in = np.log10(1 / Omega / Mu * (abs(EdivH) ** 2))
    phase_f_mat_in = np.zeros(len(EdivH))
    for i in range(len(EdivH)):
        phase_f_mat_in[i] = math.atan(EdivH[i].imag / EdivH[i].real)
    data_f = np.append(rho_f_mat_in, phase_f_mat_in)
    return  {'data_f':data_f, 'EFieldVector_in':EFieldVector_in, 'Eobs_in':Eobs_in, 'Hobs_in':Hobs_in}

def MT2DFWD2TM2(MT2DFWD2dic, gr):
    '''
    20210708
    :param MT2DFWD2dic:
    :param gr:
    :return:
    '''
    freq = MT2DFWD2dic['freq']
    Field_rho=MT2DFWD2dic['Field_rho']
    Rx = MT2DFWD2dic['Rx']
    Field_grid_x = MT2DFWD2dic['Field_grid_x']
    Field_grid_z = MT2DFWD2dic['Field_grid_z']
    X_number = MT2DFWD2dic['X_number']
    Z_number = MT2DFWD2dic['Z_number']
    Rx_index = MT2DFWD2dic['Rx_index']
    Field_sigma = 1 / 10 ** Field_rho
    Rx_z = Rx[0]
    Rx_x = Rx[1]
    Grid_air_num = 7
    eachlenz = Field_grid_z[1] - Field_grid_z[0]
    Grid_z1 = eachlenz * np.arange(-1 - Grid_air_num, 0)
    Optimal_grid_number_z2 = 30
    temp2 = np.logspace(math.log10(Field_grid_z[-1] - Field_grid_z[-2]) + 0.1, 6, Optimal_grid_number_z2)
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
    K = 1
    Epsilon = 1 / 36 / math.pi * 1e-9
    Mu = 4 * math.pi * 1e-7
    Field_sigma_temp = np.reshape(Field_sigma, (Z_number, X_number),order='f')
    Back_sigma = Field_sigma_temp[-1,:]
    Sigma = np.zeros((Nz, Nx))
    Sigma[Grid_location_z > Field_grid_z[-1],:] = np.tile(np.append(np.append(Back_sigma[0], Back_sigma), Back_sigma[-1]),
                                                          (Optimal_grid_number_z2 + 1, 1))

    # % Sigma(Grid_location_z > Field_grid_z(end),:)=Back_sigma
    # % Sigma(:, Grid_location_x < Field_grid_x(1)) = Back_sigma
    # % Sigma(:, Grid_location_x > Field_grid_x(end)) = Back_sigma
    Sigma[Surface + 1 : Surface + 1 + Z_number , 1:-1 ] = Field_sigma_temp
    Sigma[Grid_location_z < 0,:] = 1e-6
    Sigma[:,0] = Sigma[:,1]
    Sigma[:,-1] = Sigma[:,-2]
    X = np.zeros((Nz,Nx))
    Z = np.zeros((Nz,Nx))
    for j in range(Nz):
        for i in range(Nx):
            X[j,i] = Grid_location_x[i]
            Z[j,i] = Grid_location_z[j]

    deltaX = np.diff(Grid_location_x[1:])
    deltaZ = np.diff(Grid_location_z[1:])
    deltaXArray = np.zeros((Nz-2, Nx-2))
    deltaZArray = np.zeros((Nz-2,Nx-2))
    for j in range(Nz-2):
        for i in range(Nx-2):
            deltaXArray[j,i] = deltaX[i]
            deltaZArray[j,i] = deltaZ[j]
    Area = np.zeros((Nz, Nx))  # Label the solution area  first label = 1
    for ii in range(1,Nx - 1):
        Area[1: Nz - 1, ii] = np.arange((ii - 1) * (Nz - 2) + 1, (ii) * (Nz - 2)+1)
    Area = np.array(Area, dtype='int32')
    # = == == == == == == == = Make the table == == == == == == == == == == == =
    Table = np.zeros((Nxz, 2),dtype='uint32')
    q = 0
    for j in range (Nx):
        for i in range(Nz):
            if (Area[i, j] > 0):
                Table[q, 0] = i
                Table[q, 1] = j
                q = q + 1

    Rxx_index = np.array(Rx_index,dtype='uint32')
    # parfor
    # gr = 1:length(freq)
    ub = [[0+0j]*Nxz]
    Ub = np.reshape(np.array(ub),-1)
    # = == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    ja = []
    ia = [1]
    ia_temp = []
    total_count = 1
    index = 0
    value = [0j]
    Omega = 2 * math.pi * freq[gr]
    MT2ts1 = time.time()
    for i in range(Nxz):
        count = 0
        DeltaX_LEFT = X[int(Table[i, 0]), int(Table[i,1])] - X[int(Table[i,0]), int(Table[i,1]) - 1]
        DeltaZ_UP = (Z[int(Table[i,0]), int(Table[i,1])] - Z[int(Table[i,0]) - 1, int(Table[i,1])])
        DeltaZ_DOWN = (Z[int(Table[i,0]) + 1, int(Table[i,1])] - Z[int(Table[i,0]), int(Table[i,1])])
        DeltaX_RIGHT = (X[int(Table[i,0]), int(Table[i,1]) + 1] - X[int(Table[i,0]), int(Table[i,1])])
        DeltaX_CENTER = 1 / 2 * (X[int(Table[i,0]), int(Table[i,1]) + 1] - X[int(Table[i,0]), int(Table[i,1]) - 1])
        DeltaZ_CENTER = 1 / 2 * (Z[int(Table[i,0]) + 1, int(Table[i,1])] - Z[int(Table[i,0]) - 1, int(Table[i,1])])

        if (Area[int(Table[i, 0]), int(Table[i, 1]) - 1] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]), int(Table[i, 1]) - 1]-1)
            value.append(1 / DeltaX_LEFT / DeltaX_CENTER / (0.5*Sigma[Table[i,0],Table[i,1]]+ 0.5*Sigma[Table[i,0],Table[i,1]-1]-1j*Omega*Epsilon))
            index = index + 1
            count = count + 1

        if (Area[int(Table[i, 0]) - 1, int(Table[i, 1])] == 0):
            Ub[i] = -K


        if (Area[int(Table[i, 0]) - 1, int(Table[i, 1])] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]) - 1, int(Table[i, 1])]-1)
            value.append(1 / DeltaZ_UP / DeltaZ_CENTER/(0.5*Sigma[Table[i,0],Table[i,1]] +0.5*Sigma[Table[i,0]-1,Table[i,1]]-1j*Omega*Epsilon))
            index = index + 1
            count = count + 1

        if (Area[int(Table[i, 0]), int(Table[i, 1]) - 1] == 0):
            ia_temp.append(i)
            ja.append(i)
            appendee = -(1 / DeltaX_RIGHT) / DeltaX_CENTER / (0.5*Sigma[Table[i,0],Table[i,1]] +0.5*Sigma[Table[i,0],Table[i,1]+1] -1j*Omega*Epsilon) \
                       -(1/DeltaZ_DOWN/(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0]+1,Table[i,1]] -1j*Omega*Epsilon) +1/DeltaZ_UP/(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0]-1,Table[i,1]] -1j*Omega*Epsilon))/DeltaZ_CENTER \
                       + (1j*Omega*Mu)
            value.append(appendee)
                         # -(1 / DeltaZ_DOWN + 1 / DeltaZ_UP) / DeltaZ_CENTER + \
                         # (-1j * Omega * Mu * Sigma[int(Table[i, 0]), int(Table[i, 1])]+Omega**2 * Mu * Epsilon))
            index = index + 1
            count = count + 1
        elif(Area[int(Table[i, 0]), int(Table[i, 1]) + 1] == 0):
            ia_temp.append(i)
            ja.append(i)
            appendee = -(1/DeltaX_LEFT)/DeltaX_CENTER/ (0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0],Table[i,1]-1] -1j*Omega*Epsilon) \
                       -(1/DeltaZ_DOWN/(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0]+1,Table[i,1]] -1j*Omega*Epsilon) +1/DeltaZ_UP/(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0]-1,Table[i,1]] -1j*Omega*Epsilon))/DeltaZ_CENTER \
                       + (1j*Omega*Mu)
            value.append(appendee)
            index = index + 1
            count = count + 1
        else:
            ia_temp.append(i)
            ja.append(i)
            appendee=-(1/DeltaX_RIGHT/(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0],Table[i,1]+1] -1j*Omega*Epsilon)+1/DeltaX_LEFT/(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0],Table[i,1]-1] -1j*Omega*Epsilon))/DeltaX_CENTER \
                     -(1/DeltaZ_DOWN/(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0]+1,Table[i,1]] -1j*Omega*Epsilon) +1/DeltaZ_UP/(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0]-1,Table[i,1]] -1j*Omega*Epsilon))/DeltaZ_CENTER \
                     + (1j*Omega*Mu)
            value.append(appendee)
            index = index + 1
            count = count + 1


        if (Area[int(Table[i, 0]) + 1, int(Table[i, 1])] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]) + 1, int(Table[i, 1])]-1)
            value.append(1/DeltaZ_DOWN/DeltaZ_CENTER
                         /(0.5*Sigma[Table[i,0]+1,Table[i,1]]+0.5*Sigma[Table[i,0],Table[i,1]] -1j*Omega*Epsilon))
            index = index + 1
            count = count + 1

        if (Area[int(Table[i, 0]), int(Table[i, 1]) + 1] > 0):
            ia_temp.append(i)
            ja.append(Area[int(Table[i, 0]), int(Table[i, 1]) + 1]-1)
            value.append(1/DeltaX_RIGHT/DeltaX_CENTER
                         /(0.5*Sigma[Table[i,0],Table[i,1]]+0.5*Sigma[Table[i,0],Table[i,1]+1] -1j*Omega*Epsilon))
            index = index + 1
            count = count + 1

        total_count = total_count + count
        ia.append(total_count)
    MT2ts2 = time.time()
    print(multiprocessing.current_process().name)
    # print("No.{} iter".format(gr))
    print("MT2FWD equation set-up time:{} s".format(MT2ts2-MT2ts1))
    ia_temp = np.reshape(np.array(ia_temp),-1)
    ja = np.reshape(np.array(ja),-1)
    value= np.reshape(np.array(value[1:]),-1)
    D = csr_matrix((value,(ia_temp, ja)), shape = (Nxz,Nxz))
    Ux = linalg.spsolve(D,Ub)
    MT2ts3 = time.time()
    print("MT2FWD equation solving time:{} s".format(MT2ts3-MT2ts2))
    # Ux = D\Ub
    Rxz_index = math.ceil((Rx_z - Grid_location_z[0]) / eachlenz)
    HDomainArray = np.reshape(Ux, (Nz - 2, Nx - 2),order='f')
    ExFieldArray = np.zeros((Z_number,X_number),dtype='complex')
    EzFieldArray = np.zeros((Z_number,X_number),dtype='complex')
    ExArray = 1/(0.5*Sigma[1:-2,1:-1]+0.5*Sigma[2:-1,1:-1]-1j*Omega*Epsilon)*(HDomainArray[1:,:]-HDomainArray[0:-1,:])/deltaZArray[0:-1,:]
    EzArray = -1/(0.5*Sigma[1:-1,1:-2]+0.5*Sigma[1:-1,2:-1]-1j*Omega*Epsilon)* (HDomainArray[:,1:]-HDomainArray[:,0:-1])/deltaXArray[:,0:-1]

    ExFieldArray = ExArray[Surface:Surface+Z_number,:]    # in python code surface does not need to minus one
    EzFieldArray[:,0:-1] = EzArray[Surface:Surface+Z_number,:]
    HFieldArray = HDomainArray[Surface:Surface+Z_number,:]
    HFieldVector=np.reshape(HFieldArray,-1,order='f')
    ExFieldVector_in=np.reshape(ExFieldArray,-1,order='f')
    EzFieldVector_in=np.reshape(EzFieldArray,-1,order='f')
    # Rxx_index_list = np.array(Rxx_index.tolist(),dtype='int32').tolist()
    Delta_z=(Z[Surface+2,(Rxx_index).tolist()]-Z[Surface+1,(Rxx_index).tolist()])

    Hobs_in = 0.5*Ux[Area[Surface+2,(Rxx_index).tolist()]-1] + 0.5*Ux[Area[Surface+1,(Rxx_index).tolist()]-1]
    Eobs_in  = 1/(Sigma[Surface+2,(Rxx_index).tolist()]-1j*Omega*Epsilon)*(Ux[Area[Surface+3,(Rxx_index).tolist()]-1] \
    - Ux[Area[Surface+1,(Rxx_index).tolist()]-1]) / Delta_z
    # %     Rxx_index_center=round(Rx_x/eachlenx)+1
    Eobs_in = ExFieldArray[0,(Rxx_index).tolist()]
# %     EdivH = Eobs(Rxx_index)./Hobs(Rxx_index);
    EdivH = Eobs_in /Hobs_in
    rho_f_mat_in = np.log10(1 / Omega / Mu * (abs(EdivH) ** 2))
    phase_f_mat_in = np.zeros(len(EdivH))
    for i in range(len(EdivH)):
        phase_f_mat_in[i] = math.atan(EdivH[i].imag / EdivH[i].real)
    # Rxx-index minus one  surr
    data_f = np.append(rho_f_mat_in, phase_f_mat_in)
    return  {'data_f':data_f, 'ExFieldVector_in':ExFieldVector_in, 'EzFieldVector_in':EzFieldVector_in,'Eobs_in':Eobs_in, 'Hobs_in':Hobs_in}

# [data_f,EobsVector,HobsVector,ExFieldVector,EzFieldVector]

import tensorflow as tf
from tensorflow.keras import backend as K
def MT2DFWD2_zhhy_tf(MT2DFWD2dic, gr):
    freq = MT2DFWD2dic['freq']
    Field_rho = MT2DFWD2dic['Field_rho']
    Rx = MT2DFWD2dic['Rx']
    Field_grid_x = MT2DFWD2dic['Field_grid_x']
    Field_grid_z = MT2DFWD2dic['Field_grid_z']
    X_number = MT2DFWD2dic['X_number']
    Z_number = MT2DFWD2dic['Z_number']
    Rx_index = MT2DFWD2dic['Rx_index']
    ia_temp = MT2DFWD2dic['ia_temp']
    ja = MT2DFWD2dic['ja']
    value = MT2DFWD2dic['value']
    Ub = MT2DFWD2dic['Ub']
    Area = MT2DFWD2dic['Area']
    index1 = MT2DFWD2dic['index1']
    Z = MT2DFWD2dic['Z']
    Rxx_index = Rx_index
    Field_sigma = 1 / 10 ** Field_rho
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
    # print("2")
    # Field_sigma_temp = tf.reshape(Field_sigma, (Z_number, X_number))
    Field_sigma_temp = Field_sigma[:Z_number, :]
    for qq in range(1, X_number):
        Field_sigma_temp = tf.concat((Field_sigma_temp, Field_sigma[qq * Z_number:(qq + 1) * Z_number, :]), axis=1)
    Back_sigma = Field_sigma_temp[-1:, :]
    # Sigma = tf.zeros((Nz, Nx))
    # Sigma_concat1 = np.zeros((Surface+1, X_number))+1e-6
    # Sigma_concat11 = tf.constant(Sigma_concat1, dtype=tf.float32)
    Sigma_concat11 = Field_sigma_temp[:Surface + 1, :] * 0 + 1e-6
    Sigma_concat2 = Field_sigma_temp
    Sigma_concat3 = Back_sigma
    for gg in range(Optimal_grid_number_z2):
        Sigma_concat3 = tf.concat((Sigma_concat3, Back_sigma), axis=0)
    # Sigma_concat3 = tf.experimental.numpy.tile(Back_sigma, (Optimal_grid_number_z2 + 1, 1))
    # Sigma = Sigma[Grid_location_z > Field_grid_z[-1], :].assign(tf.experimental.numpy.tile(
    #     tf.experimental.numpy.append(tf.experimental.numpy.append(Back_sigma[0], Back_sigma), Back_sigma[-1]),
    #     (Optimal_grid_number_z2 + 1, 1)))
    # Sigma[Surface + 1: Surface + 1 + Z_number, 1:-1] = Field_sigma_temp
    # Sigma[Grid_location_z < 0, :] = 1e-6
    Sigma = tf.concat([Sigma_concat11, Sigma_concat2, Sigma_concat3], axis=0)
    # Sigma_active = tf.reshape(Sigma[1:-1, :], -1)   ####这里不对 应该是'f'
    Sigma_active = Sigma[1:-1, 0]
    for pp in range(1,Sigma.shape[1]):
        Sigma_active = tf.concat((Sigma_active, Sigma[1:-1,pp]), axis=0)
    MT2ts2 = time.time()
    Omega = 2 * math.pi * freq[gr]
    Ub1 = np.zeros(len(Ub), dtype='complex')
    Ub1[Ub == 1] = tf.cast(tf.complex(0.0,1.0), dtype=tf.complex128) * K.eval(Omega) * Mu * J
    Ub1_tf = tf.constant(Ub1, shape=(len(Ub1), 1))
    value1 = value.copy()
    value1 = tf.constant(value1)
    temp1 = 0j * value1[:index1]
    temp2 = 0j * value1[index1:] -1j * K.eval(Omega) * Mu * tf.cast(Sigma_active, dtype=tf.complex128) + K.eval(Omega) ** 2 * Mu * Epsilon
    temp = tf.concat((temp1, temp2), axis=0)
    value1 = value1 + temp
    # value1[index1:] = value1[index1:] + -1j * Omega * Mu * Sigma_active
    # value1[index1:] = value1[index1:] + Omega ** 2 * Mu * Epsilon
    # D = csr_matrix((value1, (ia_temp, ja)), shape=(len(Sigma_active), len(Sigma_active)))
    indices = []
    for pp in range(len(ia_temp)):
        indices.append([ia_temp[pp], ja[pp]])
    D_tf_real=tf.sparse.SparseTensor(indices, tf.math.real(value1), [len(Sigma_active), len(Sigma_active)])
    D_tf_imag = tf.sparse.SparseTensor(indices, tf.math.imag(value1), [len(Sigma_active), len(Sigma_active)])
    D_tf_real = tf.sparse.reorder(D_tf_real)
    D_tf_r = tf.sparse.to_dense(D_tf_real)
    D_tf_imag = tf.sparse.reorder(D_tf_imag)
    D_tf_i = tf.sparse.to_dense(D_tf_imag)
    D_tf = tf.cast(D_tf_r, dtype=tf.complex128)+tf.cast(tf.complex(0.0,1.0), dtype=tf.complex128)*tf.cast(D_tf_i, dtype=tf.complex128)
    # D_tf = tf.cast(D_tf)
    Ux = tf.linalg.solve(D_tf, Ub1_tf)
    MT2ts3 = time.time()
    # print("MT2FWD equation solving time:{} s".format(MT2ts3 - MT2ts2))
    # Ux = D\Ub
    Rxz_index = math.ceil((Rx_z - Grid_location_z[0]) / eachlenz)
    EDomainArray = np.reshape(K.eval(Ux), (Nz - 2, Nx - 2), order='f')
    EFieldArray = EDomainArray[Surface:Surface + Z_number, :]
    EFieldVector_in = np.reshape(EFieldArray, -1, order='f')
    Rxx_index_list = np.array(Rxx_index.tolist(), dtype='int64').tolist()
    # Ux_np = K.eval(Ux)

    list_tmp1 = (Area[Surface+1, Rxx_index_list] - 1).tolist()
    Eobs_in = Ux[list_tmp1[0], :]
    for pp in range(1, len(list_tmp1)):
        Eobs_in = tf.concat((Eobs_in, Ux[list_tmp1[pp], :]), axis=0)
    Eobs_in = tf.reshape(Eobs_in, (-1,1))
    # Eobs_in = Ux_np[(Area[Surface + 1, Rxx_index_list] - 1).tolist()]
    Delta_z = Z[Surface + 1, Rxx_index_list] - Z[Surface, Rxx_index_list]
    Delta_z_tf = tf.constant(Delta_z)
    Delta_z_tf = tf.reshape(Delta_z_tf, (-1, 1))
    list_tmp2 = Area[Surface + 2, Rxx_index_list] - 1
    list_tmp3 = Area[Surface, Rxx_index_list] - 1
    H_tmp1 = Ux[list_tmp2[0]]-Ux[list_tmp3[0]]
    for pp in range(1, len(list_tmp2)):
        H_tmp1 = tf.concat((H_tmp1, Ux[list_tmp2[pp]] - Ux[list_tmp3[pp]]), axis=0)
    H_tmp1 = tf.reshape(H_tmp1, (-1,1))
    Hobs_in = -1 / tf.cast(tf.complex(0.0,1.0), dtype=tf.complex128) / K.eval(Omega) / Mu * (
            tf.cast(H_tmp1, dtype=tf.complex128)) / 2 / tf.cast(Delta_z_tf, dtype=tf.complex128)
    # % Rxx_index_center = round(Rx_x / eachlenx) + 1
    # % EdivH = Eobs(Rxx_index). / Hobs(Rxx_index)
    EdivH = tf.cast(Eobs_in, dtype=tf.complex128) / Hobs_in
    rho_f_mat_in = tf.experimental.numpy.log10(1 / K.eval(Omega) / Mu * (abs(EdivH) ** 2))
    # phase_f_mat_in = tf.zeros(len(EdivH))
    # for i in range(len(EdivH)):
    phase_f_mat_in = tf.math.atan(tf.math.imag(EdivH) / tf.math.real(EdivH))
    # rho_f_mat_in = tf.concat()
    data_f = tf.experimental.numpy.append(rho_f_mat_in, phase_f_mat_in)
    # print("Single frequency FWD finished!")
    return {'data_f': data_f, 'EFieldVector_in': EFieldVector_in, 'Eobs_in': Eobs_in, 'Hobs_in': Hobs_in}
    # return {'data_f': Field_sigma}

def MT2DFWD2_zhhy_tf_4_batch(MT2DFWD2dic, gr):
    freq = MT2DFWD2dic['freq']
    Field_rho = MT2DFWD2dic['Field_rho']
    Rx = MT2DFWD2dic['Rx']
    Field_grid_x = MT2DFWD2dic['Field_grid_x']
    Field_grid_z = MT2DFWD2dic['Field_grid_z']
    X_number = MT2DFWD2dic['X_number']
    Z_number = MT2DFWD2dic['Z_number']
    Rx_index = MT2DFWD2dic['Rx_index']
    ia_temp = MT2DFWD2dic['ia_temp']
    ja = MT2DFWD2dic['ja']
    value = MT2DFWD2dic['value']
    Ub = MT2DFWD2dic['Ub']
    Area = MT2DFWD2dic['Area']
    index1 = MT2DFWD2dic['index1']
    Z = MT2DFWD2dic['Z']
    Rxx_index = Rx_index
    Field_sigma = 1 / 10 ** Field_rho
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
    # print("2")
    # Field_sigma_temp = tf.reshape(Field_sigma, (Z_number, X_number))
    Field_sigma_temp = Field_sigma[:Z_number, :]
    for qq in range(1, X_number):
        Field_sigma_temp = tf.concat((Field_sigma_temp, Field_sigma[qq * Z_number:(qq + 1) * Z_number, :]), axis=1)
    Back_sigma = Field_sigma_temp[-1:, :]
    # Sigma = tf.zeros((Nz, Nx))
    # Sigma_concat1 = np.zeros((Surface+1, X_number))+1e-6
    # Sigma_concat11 = tf.constant(Sigma_concat1, dtype=tf.float32)
    Sigma_concat11 = Field_sigma_temp[:Surface + 1, :] * 0 + 1e-6
    Sigma_concat2 = Field_sigma_temp
    Sigma_concat3 = Back_sigma
    for gg in range(Optimal_grid_number_z2):
        Sigma_concat3 = tf.concat((Sigma_concat3, Back_sigma), axis=0)
    # Sigma_concat3 = tf.experimental.numpy.tile(Back_sigma, (Optimal_grid_number_z2 + 1, 1))
    # Sigma = Sigma[Grid_location_z > Field_grid_z[-1], :].assign(tf.experimental.numpy.tile(
    #     tf.experimental.numpy.append(tf.experimental.numpy.append(Back_sigma[0], Back_sigma), Back_sigma[-1]),
    #     (Optimal_grid_number_z2 + 1, 1)))
    # Sigma[Surface + 1: Surface + 1 + Z_number, 1:-1] = Field_sigma_temp
    # Sigma[Grid_location_z < 0, :] = 1e-6
    Sigma = tf.concat([Sigma_concat11, Sigma_concat2, Sigma_concat3], axis=0)
    # Sigma_active = tf.reshape(Sigma[1:-1, :], -1)   ####这里不对 应该是'f'
    Sigma_active = Sigma[1:-1, 0]
    for pp in range(1,Sigma.shape[1]):
        Sigma_active = tf.concat((Sigma_active, Sigma[1:-1,pp]), axis=0)
    MT2ts2 = time.time()
    freq_np = K.eval(freq)
    print('In forward problem, fre = {}'.format(freq_np))
    Omega = 2 * math.pi * freq_np[gr]
    Ub1 = np.zeros(len(Ub), dtype='complex')
    Ub1[Ub == 1] = np.complex(0.0,1.0) * Omega * Mu * J
    Ub1_tf = tf.constant(Ub1, shape=(len(Ub1), 1))
    value1 = value.copy()
    value1 = tf.constant(value1)
    temp1 = 0j * value1[:index1]
    temp2 = 0j * value1[index1:] -1j * K.eval(Omega) * Mu * tf.cast(Sigma_active, dtype=tf.complex128) + K.eval(Omega) ** 2 * Mu * Epsilon
    temp = tf.concat((temp1, temp2), axis=0)
    value1 = value1 + temp
    # value1[index1:] = value1[index1:] + -1j * Omega * Mu * Sigma_active
    # value1[index1:] = value1[index1:] + Omega ** 2 * Mu * Epsilon
    # D = csr_matrix((value1, (ia_temp, ja)), shape=(len(Sigma_active), len(Sigma_active)))
    indices = []
    for pp in range(len(ia_temp)):
        indices.append([ia_temp[pp], ja[pp]])
    tmpg = Sigma_active.shape[0]
    D_tf_real=tf.sparse.SparseTensor(indices, tf.math.real(value1), [tmpg, tmpg])
    D_tf_imag = tf.sparse.SparseTensor(indices, tf.math.imag(value1), [tmpg, tmpg])
    D_tf_real = tf.sparse.reorder(D_tf_real)
    # D_tf_r = tf.sparse.to_dense(D_tf_real)
    D_tf_imag = tf.sparse.reorder(D_tf_imag)
    # D_tf_i = tf.sparse.to_dense(D_tf_imag)
    D_tf = tf.cast(tf.sparse.to_dense(D_tf_real), dtype=tf.complex64)\
           +tf.complex(0.0,1.0)*tf.cast(tf.sparse.to_dense(D_tf_imag), dtype=tf.complex64)
    # D_tf = tf.constant(D.toarray())
    Ux = tf.linalg.solve(D_tf, tf.cast(Ub1_tf, dtype=tf.complex64))
    MT2ts3 = time.time()
    # print("MT2FWD equation solving time:{} s".format(MT2ts3 - MT2ts2))
    Rxz_index = math.ceil((Rx_z - Grid_location_z[0]) / eachlenz)
    EDomainArray = tf.transpose(tf.reshape(Ux, (Nx - 2, Nz - 2)))
    EFieldArray = EDomainArray[Surface:Surface + Z_number, :]
    EFieldVector_in = tf.reshape(tf.transpose(EFieldArray), (-1,))
    Rxx_index_list = np.array(Rxx_index.tolist(), dtype='int32').tolist()
    # Ux_np = K.eval(Ux)

    list_tmp1 = (Area[Surface, Rxx_index_list] - 1).tolist()
    Eobs_in = Ux[list_tmp1[0], :]
    for pp in range(1, len(list_tmp1)):
        Eobs_in = tf.concat((Eobs_in, Ux[list_tmp1[pp], :]), axis=0)
    Eobs_in = tf.reshape(Eobs_in, (-1,1))
    # Eobs_in = Ux_np[(Area[Surface + 1, Rxx_index_list] - 1).tolist()]
    Delta_z = Z[Surface + 1, Rxx_index_list] - Z[Surface, Rxx_index_list]
    Delta_z_tf = tf.constant(Delta_z)
    Delta_z_tf = tf.reshape(Delta_z_tf, (-1, 1))
    list_tmp2 = Area[Surface + 1, Rxx_index_list] - 1
    list_tmp3 = Area[Surface-1, Rxx_index_list] - 1
    H_tmp1 = Ux[list_tmp2[0]]-Ux[list_tmp3[0]]
    for pp in range(1, len(list_tmp2)):
        H_tmp1 = tf.concat((H_tmp1, Ux[list_tmp2[pp]] - Ux[list_tmp3[pp]]), axis=0)
    H_tmp1 = tf.reshape(H_tmp1, (-1,1))
    Hobs_in = -1 / tf.cast(tf.complex(0.0,1.0), dtype=tf.complex128) / K.eval(Omega) / Mu * (
            tf.cast(H_tmp1, dtype=tf.complex128)) / 2 / tf.cast(Delta_z_tf, dtype=tf.complex128)
    # % Rxx_index_center = round(Rx_x / eachlenx) + 1
    # % EdivH = Eobs(Rxx_index). / Hobs(Rxx_index)
    EdivH = tf.cast(Eobs_in, dtype=tf.complex128) / Hobs_in
    rho_f_mat_in = tf.experimental.numpy.log10(1 / K.eval(Omega) / Mu * (abs(EdivH) ** 2))
    # phase_f_mat_in = tf.zeros(len(EdivH))
    # for i in range(len(EdivH)):
    phase_f_mat_in = tf.math.atan(tf.math.imag(EdivH) / tf.math.real(EdivH))
    # rho_f_mat_in = tf.concat()
    data_f = tf.experimental.numpy.append(rho_f_mat_in, phase_f_mat_in)
    # print("Single frequency FWD finished!")
    return {'data_f': data_f, 'EFieldVector_in': EFieldVector_in, 'Eobs_in': Eobs_in, 'Hobs_in': Hobs_in}
    # return {'data_f': Field_sigma}

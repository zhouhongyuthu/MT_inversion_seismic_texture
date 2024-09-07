# 20240906 Github Opensource
# A copy of  NLCG_MT_pix-bas_seism_style_for-Marm_230226-Use.py
# -*- coding: utf-8 -*-
import numpy as np
# from skimage.io import imread
import matplotlib.pyplot as plt
import time
import math
import scipy.io as io
import multiprocessing
import functools
import numpy.linalg as lg
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.sparse import csr_matrix
from tensorflow.keras import backend as K
import os
from scipy.io import savemat, loadmat
from scipy.ndimage import filters
import helpers_SsM as h_SsM
import tensorflow as tf
import MT2DFWD2 as MT
import platform
# from skimage.metrics import structural_similarity as ssim
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
# import helpers as helpers
import Jacobians as Jacos


def style_function(aS,neural_transfer_model, image, if_res = 1, nor = [0, 0]):
    # if_res == 0:输入的是地震叠后数据，=1：输入的是
    # nor：当输入是增量电阻率图像时，自定义输入到神经网络中的normalization系数
    ### 

    from tensorflow.keras import backend as K
    # image = tf.cast(image, dtype=tf.double)
    # image1 = tf.cast(image, dtype=tf.float64)
    # MS_outputs = []
    MS_outputs = tf.constant([0.0, 0.0])
    MS_outputs = tf.cast(MS_outputs, dtype=tf.float64)

    image_max = K.eval(tf.convert_to_tensor(tf.math.reduce_max(image)))
    image_min = K.eval(tf.convert_to_tensor(tf.math.reduce_min(image)))

    # if (if_res):
        # print("*==============*")
        # print("In seismic style function: before: image min={}, max={}".format(image_min, image_max))

    # image1 = (image - tf.constant(1, dtype=tf.float64))/(tf.constant(2, dtype=tf.float64))   # 归一化，能成吗？
    if(if_res == 0):
        image = (image - tf.constant(image_min, dtype=tf.float64)) / (tf.constant(image_max - image_min + 1e-9, dtype=tf.float64))  # 归一化，能成吗？
    else:
        imi1 = image_min + 1E-9
        ima1 = image_max + 1E-9
        image = (image - tf.constant(imi1, dtype=tf.float64)) / (tf.constant(ima1 - imi1, dtype=tf.float64))
    image_max = K.eval(tf.convert_to_tensor(tf.math.reduce_max(image)))
    image_min = K.eval(tf.convert_to_tensor(tf.math.reduce_min(image)))

    print("In seismic style function: after: image min={:.3e}, max={:.3e}".format(image_min, image_max))

    for ii in range(len(neural_transfer_model)):
        neural_transfer_model_ii = neural_transfer_model[ii]
        style_output = neural_transfer_model_ii(image)  ## 直接输出还是float32
        style_output = tf.cast(style_output, dtype=tf.float64)
        # extract the content layer output and style layer output
        # style_feature = [layer for layer in style_output]
        # grams = []tf.shape(style_output)
        sp = tf.shape(style_output)
        sp = K.eval(sp)
        MS1 = tf.zeros(int(sp[3]))
        MS2 = tf.zeros(int(sp[3]))
        sz = int(sp[2])
        thick = int(sp[3])
        # for style_output_1 in style_output:
        style_output_1_re = tf.reshape(style_output, (sz * sz, thick))
        MS1 = tf.math.reduce_mean(style_output_1_re, axis=0)
        MS2 = tf.math.reduce_std(style_output_1_re, axis=0)
        # gram_output = gram_matrix(output_layer)
        # grams.append(gram_output)
        # gram_output = tf.reshape(gram_output, -1)
        # MS_outputs.append(MS1)
        # MS_outputs.append(MS2)
        MS_output1 = tf.concat([MS1, MS2], 0)
        aS_i = tf.constant(np.sqrt(aS[ii])+1E-12)
        aS_i = tf.cast(aS_i, dtype=tf.float64)
        MS_outputs = tf.concat([MS_outputs, MS_output1], 0) * aS_i
    return MS_outputs[2:]  # 矩阵 [特征图组数*2,...]

def get_Allpix_2_Parpix(input, value):
    # input: 224*224, 方阵
    # value：input中的目标数值
    ## 根据地震结构 设置属性分配矩阵 得到224*224图，输入等于value的像素为1，其他像素为0
    ### 
    No1_map = np.array((input==value), dtype=np.float64)
    No1_map = np.reshape(No1_map, (224, 224), order='f')
    # No1_map_squ = csr_matrix(np.diag(No1_map))
    w1 = h_SsM.fspecial_gaussian(np.array([6, 6]), 2)
    No1_map = filters.convolve(No1_map, w1)
    return No1_map

def get_pix_std_constraint(x_var_res_f, Weis, ZNXN):
    # x_var_res_f：只约束像素部分
    # Weis: [a,b]分别是均值的权重和方差的权重
    # x_var_res_f = x_var[ZNXN:]
    h_constraint = Weis[0]*tf.square(tf.math.reduce_mean(x_var_res_f))  # 均值的平方
    h_constraint = h_constraint + Weis[1]*tf.math.reduce_std(x_var_res_f)
    return h_constraint

def get_hybrid_style_constraint(image_seis, image_res):
    ## num: 取01234567
    ### image_seis[1, 224*2, 224*4, 1]
    ## image_res：[1, 224*2, 224*4, 1]

    style_cost = tf.constant(0, dtype=tf.float64)
    bs = 2
    # 单个长度是180
    patch_len = 112
    for pp in range(2):  # 4
        for qq in range(2):  # 8
            No1_map = np.zeros((1, 112*2, 112*2, 1))
            z_max = (pp + 1) * patch_len
            x_max = (qq + 1) * patch_len
            # if (pp == 1):
            #     z_max = 448
            # if (qq == 3):
            #     x_max = 896
            No1_map[:, pp*patch_len:z_max, qq*patch_len:x_max, :] = 1
            single_fil_len = 2
            w1 = h_SsM.fspecial_gaussian(np.array([single_fil_len*2+1, single_fil_len*2+1]), 2)
            No1_map[0,:,:,0] = filters.convolve(No1_map[0,:,:,0], w1)

            No1_map = tf.constant(No1_map, dtype=tf.float64)
            image_seis_pp = image_seis * No1_map  # [1,448,896,1]
            image_res_pp = image_res * No1_map  # [1,448,896,1]

            f_ground = style_function(ahphaStyle, neural_transfer_model, image_seis_pp, if_res=0)
            f_res = style_function(ahphaStyle, neural_transfer_model, image_res_pp, if_res=1, nor=[-13, 13])
            style_cost = style_cost + tf.norm(f_res - f_ground, 2) ** 2
    return style_cost

def computeGradient(Field_grid_x, Field_grid_z, X_number, Z_number):
    gridNumber = X_number * Z_number
    iVertical = -1 * np.ones(2 * gridNumber)
    jVertical = -1 * np.ones(2 * gridNumber)
    valueVertical = -10 * np.ones(2 * gridNumber)
    iHorizontal = -1 * np.ones(2 * gridNumber)
    jHorizontal = -1 * np.ones(2 * gridNumber)
    valueHorizontal = -10 * np.ones(2 * gridNumber)
    indexVertical = 0
    indexHorizontal = 0
    for i in range(gridNumber):
        temp1 = i % Z_number
        if temp1 != Z_number - 1:
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
    valueVertical = valueVertical[valueVertical != -10]
    iHorizontal = iHorizontal[iHorizontal != -1]
    jHorizontal = jHorizontal[jHorizontal != -1]
    valueHorizontal = valueHorizontal[valueHorizontal != -10]
    verticalGradient = csr_matrix((valueVertical, (iVertical, jVertical)), shape=(gridNumber, gridNumber))
    horizontalGradient = csr_matrix((valueHorizontal, (iHorizontal, jHorizontal)), shape=(gridNumber, gridNumber))
    return verticalGradient, horizontalGradient

def get_property_from_code_pixel(x_var, ZNXN):
    x_var1 = K.eval(tf.convert_to_tensor(x_var))
    tmp1 = np.reshape(x_var1[:ZNXN], (ZNumberMT, XNumberMT), order='f')
    # tmp1 = tmp1[:, Number_yantuo:-Number_yantuo]
    intp2d = rbs(zElementLocationMT, xElementLocationMT, tmp1,
                 [min(min(zElementLocationMT), min(zElementLocationPSFiTe)) - 5,
                  max(max(zElementLocationMT), max(zElementLocationPSFiTe)) + 5,
                  min(min(xElementLocationMT), min(xElementLocationPSFiTe)) - 5,
                  max(max(xElementLocationMT), max(xElementLocationPSFiTe)) + 5])
    x_var_qiann = intp2d(zElementLocationPSFiTe, xElementLocationPSFiTe)

    x_var_qiann = np.reshape(x_var_qiann, -1, order='f')

    x_var_r = x_var_qiann + x_var1[ZNXN:]

    return x_var_r, x_var_qiann, x_var1[ZNXN:]

@tf.custom_gradient
def custom_op(x_var_input):  # , d_ground_ii1, frequencyMT_tf1

    MT2DFWD2_packet = {'freq': frequencyMT, 'Field_rho': 0, 'Rx': RxMT, 'Field_grid_x': xElementLocationMT,
                       'Field_grid_z': zElementLocationMT, 'X_number': XNumberMT, 'Z_number': ZNumberMT,
                       'Rx_index': rxIndexMT}

    if SELECT_AS_INVERSIONMODE:
        pool = multiprocessing.Pool(8)

    MT2DFWD2_packet['ia_temp'] = ia_temp
    MT2DFWD2_packet['ja'] = ja
    MT2DFWD2_packet['value'] = value
    MT2DFWD2_packet['Ub'] = Ub
    MT2DFWD2_packet['Area'] = Area
    MT2DFWD2_packet['index1'] = index1
    MT2DFWD2_packet['Z'] = Z

    tm1 = time.time()
    MT2DFWD2_packet['Field_rho'] = x_var_tmp
    if SELECT_AS_INVERSIONMODE:
        MT2DFWD2_back = pool.map(functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_packet), range(len(frequencyMT)))
    else:
        MT2DFWD2_back = []
        for ii in range(len(frequencyMT)):
            bci = MT.MT2DFWD2_zhhy(MT2DFWD2_packet, ii)
            MT2DFWD2_back.append(bci)

    EFieldVectorf = np.zeros((XNumberMT * ZNumberMT, len(frequencyMT)), dtype='complex')
    EobsVector = np.zeros(len(rxIndexMT) * len(frequencyMT), dtype='complex')
    HobsVector = np.zeros(len(rxIndexMT) * len(frequencyMT), dtype='complex')
    newAppResist = np.zeros(len(rxIndexMT) * 2 * len(frequencyMT))

    for i in range(len(frequencyMT)):
        newAppResist[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][:len(rxIndexMT)]
        newAppResist[len(rxIndexMT) * len(frequencyMT) + i * len(rxIndexMT):len(rxIndexMT) * len(frequencyMT) + (i + 1) * len(
            rxIndexMT)] = MT2DFWD2_back[i]['data_f'][len(rxIndexMT):]
        EobsVector[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['Eobs_in']
        HobsVector[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['Hobs_in']
        EFieldVectorf[:, i] = MT2DFWD2_back[i]['EFieldVector_in']

    d_obs = tf.constant(newAppResist, dtype=tf.float64)

    result = alphaData * tf.cast((tf.norm(d_obs - d_ground, 2)/ tf.norm(d_ground, 2)) ** 2, dtype=tf.float64)

    def custom_grad(dy):

        [jacobianMTRho, jacobianMTPhi] = Jacos.ComputeJacobianFunc_z(EobsVector, HobsVector, EFieldVectorf, frequencyMT,
                                                                     x_var_tmp, RxMT, xElementLocationMT,
                                                                     zElementLocationMT, XNumberMT, ZNumberMT,
                                                                     rxIndexMT, ia_temp, ja, value, Area, index1)

        jacobianMT = np.concatenate((jacobianMTRho, jacobianMTPhi), axis=0)

        jacobianMT = K.eval(jacobianMT)

        grad = -2*np.dot(jacobianMT.T, K.eval(d_ground-d_obs))/np.linalg.norm(K.eval(d_ground))**2

        print("sum delta data = {}".format(np.sum(K.eval(d_ground-d_obs))))

        ## 
        grad2 = -2*np.dot(np.dot(jacob_xi_2_yantuo, jacobianMT.T), K.eval(d_ground-d_obs))/np.linalg.norm(K.eval(d_ground))**2

        grad0 = np.concatenate((grad, grad2), axis=0)

        return tf.convert_to_tensor(grad0, dtype=tf.double)
    return result, custom_grad

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def call(self, x):
        return custom_op(x)  # you don't need to explicitly define the custom gradient
                             # as long as you registered it with the previous method

if __name__ == '__main__':
    ### 
    tm1 = time.time()
    GPUNo = 3
    sys = platform.system()
    if sys == "Windows":
        print("OS is Windows.")
    elif sys == "Linux":
        # plt.switch_backend('Agg')
        print("OS is Linux.")
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=gpus[GPUNo], device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpus[GPUNo], enable=True)

    bc = [4]
    style_layers = []
    for ii in range(len(bc)):
        style_layers.append('block' + str(bc[ii]) + '_conv1')
    number_style = len(style_layers)
    layer_num = ['all']  # Gram矩阵维度：layer_num*layer_num

    NP = 1
    style_UID = '002'
    init_list1 = [19.42934712346477, 67.34095449500555, 41.05050956552833, 59.56746000159197, 40.88857466063348,
                  44.811446740858514, 49.24116449023602, 57.5414543770264, 26.094806686231976, 51.854586799114344,
                  86.65616170397739,
                  26.638402518197914, 21.95508021390374, 41.79063409471734, 23.777347407848815, 34.22433783640037,
                  81.98945109958876,
                  44.67916646427357, 82.97854671280277, 53.629198804559856]
    init_list = np.array(init_list1) + np.random.rand() * 10 - 5
    initResis = 150
    alphaVerMT = 10E-4 # 10E-4 14E-4
    alphaHorMT = 5E-4 #  5E-4 7E-4
    alphaVerMT3 = 0E-8
    alphaHorMT3 = 0E-8
    alphaData = 1
    ahphaStyle = [1e-8]  # 
    # ↑
    # Weis = [1e-1, 8e-5]
    Weis = [1e-2, 2e-8]   # layer4:2E-8
    # Weis = [1e-2, 5e-5]
    MaxN = 161
    l1nm = 2 # 0 均值方差正则化  1 1范数 2 2范数

    rsse = 0.06  ## 范围0-1， =1则seismic原图；=0则seismic rescale为0
    trmono = 0
    TB = -1
    fre1 = 0 #-2.5
    fre2 = 3.5 #2.5
    freqNumberMT = 22
    rxNumberMT = 16
    bar_1 = 1
    bar_2 = 2
    SELECT_SAVE = 1  # 0 or 1. 0-don't save intermediate result and images. 1-save. can just keep 0
    SELECT_AS_INVERSIONMODE = 0  # 0 or 1. 0-do not use parallelism | 1-use parallelism
    sel_norm = 'l2'
    use_MTo_bef_x = 0
    use_MTo_as_init = 2  # 
    Nside = 5   # 

    ssit = 0.4   ## 地震初始模型范围[-ssit, ssit]，如0.3 0.4
    MTit = 1  ## 用数据插值时，初始模型在数据基础上的系数

    learning_rate1 = 25e-2
    learning_rate2 = 25e-2

    XNumberMT = 50  # 50#100#120
    ZNumberMT = 64  # 64#80#120

    UID = '[ae]sy_Mm_pb_aV_{}_aH_{}_aS_{}_Ws-{}_bc-{}_uMai-{}_MTit_{}_ssit_{}-MTG-SMrg' \
          '-Yz({}{})-fre1-{}-SSDZ2'.\
        format(alphaVerMT, alphaHorMT,ahphaStyle, Weis, bc, use_MTo_as_init, MTit,
               ssit, ZNumberMT, XNumberMT, fre1)

    if SELECT_SAVE:
        if not os.path.exists('./GN_inversion/'+UID):
            os.makedirs('./GN_inversion/'+UID)
            print('./GN_inversion/'+UID+' does not exist! create it.')

# %%
    if SELECT_AS_INVERSIONMODE:
        pool = multiprocessing.Pool(16)
    # unchangable parameters: needn't to change generally
    alphaBetMT = 0.1

    RmsC_MT = 0.006

    print("Mesh Set-Up...")
    timestamp1 = time.time()
    # XNumberSMC = 150

    fieldXStart = 0
    fieldXEnd = 9200

    xEdgeLocationMT = np.linspace(fieldXStart, fieldXEnd, XNumberMT + 1)
    xElementLocationMT = 0.5 * (xEdgeLocationMT[0:-1] + xEdgeLocationMT[1:])

    dh = np.zeros(ZNumberMT)
    for i in range(ZNumberMT):
        dh[i] = 15 * math.pow(1.032, i)
        # dh[i] = 8 * math.pow(1.033, i)
        # dh[i] = 6 * math.pow(1.020, i)
    zEdgeLocationMT = np.concatenate(([0],np.cumsum(dh)))
    zElementLocationMT = 0.5 * (zEdgeLocationMT[0:-1] + zEdgeLocationMT[1:])
    [xMT, yMT] = np.meshgrid(xElementLocationMT, -zElementLocationMT)
    [xElementLengthMT, zElementLengthMT] = np.meshgrid(np.diff(xEdgeLocationMT), np.diff(zEdgeLocationMT))
    elementSizeMT = xElementLengthMT * zElementLengthMT
    gridNumberMT = ZNumberMT * XNumberMT
    timestamp2 = time.time()
    domainDepth = zEdgeLocationMT[-1]

    XNumberPSFiTe = 224
    xEdgeLocationPSFiTe = np.linspace(fieldXStart, fieldXEnd, XNumberPSFiTe + 1)
    xElementLocationPSFiTe = 0.5 * (xEdgeLocationPSFiTe[0:-1] + xEdgeLocationPSFiTe[1:])
    ZNumberPSFiTe = 224
    zEdgeLocationPSFiTe = np.linspace(0, domainDepth, ZNumberPSFiTe + 1)
    zElementLocationPSFiTe = 0.5 * (zEdgeLocationPSFiTe[0:-1] + zEdgeLocationPSFiTe[1:])
    [xPSFiTe, yPSFiTe] = np.meshgrid(xElementLocationPSFiTe, -zElementLocationPSFiTe)
    [xElementLengthPS, zElementLengthPS] = np.meshgrid(np.diff(xEdgeLocationPSFiTe), np.diff(zEdgeLocationPSFiTe))
    elementSizePS = xElementLengthPS * zElementLengthPS
    gridNumberPS = ZNumberPSFiTe * XNumberPSFiTe

    # F2C = loadmat('fine_2_coar-224224-6450_spa_v2.mat')  ## 
    # F2C = loadmat('fine_2_coar-224224-6450_spa.mat')  ## 
    F2C = loadmat('fine_2_coar-224224-{}{}_spa.mat'.format(ZNumberMT, XNumberMT))  ##
    fine_2_coar = F2C['fine_2_coar_sparse']  # numpy array  【XN精细*ZN精细，XN粗糙*ZN粗糙】
    fine_2_coar = np.array(fine_2_coar.todense(), dtype=np.float64)
    fine_2_coar = np.transpose(fine_2_coar)  # 【粗糙，精细】
    fine_2_coar = tf.constant(fine_2_coar, dtype=tf.float64)

    InvParas = [alphaVerMT, alphaHorMT, alphaBetMT]
    reference_listR = []
    result_listR = []

    frequencyMT = np.logspace(fre1, fre2, freqNumberMT)
    rxIndexMT = np.array(np.linspace(5, XNumberMT-5, rxNumberMT), dtype='int32')
    rxNumberMT = len(rxIndexMT)
    RxMT = [0, 10000]

    model_part = loadmat("Final_block_224224-20.mat")  # 0-31
    true_rho_cat = model_part['labels3']
    ### 

    # ##### 
    true_rho_cat1 = np.zeros((ZNumberPSFiTe, XNumberPSFiTe))   # 0-63

    grid_sz = int(np.round(ZNumberPSFiTe/Nside))  # 28
    for pp in range(Nside):
        for oo in range(Nside):
            if(pp==Nside-1 or oo == Nside-1):
                true_rho_cat1[pp * grid_sz:ZNumberPSFiTe, oo * grid_sz:XNumberPSFiTe] = pp * Nside + oo
            else:
                true_rho_cat1[pp*grid_sz:(pp+1)*grid_sz, oo*grid_sz:(oo+1)*grid_sz] = pp*Nside+oo
    N_seism_corr = Nside*Nside
    # #####

    ####
    # true_rho_cat1 = true_rho_cat.copy()
    ###
    true_rho_cat1 = np.reshape(true_rho_cat1, -1, order='f')
    diff_atts = np.unique(true_rho_cat1)
    # N_seism_corr = len(diff_atts)

    resis_model = loadmat('Marmousi.mat')
    xs = np.array(resis_model['xs'], dtype=np.float64)
    zs = np.array(resis_model['zs'], dtype=np.float64)
    image = np.array(resis_model['model'], dtype=np.float64)
    true_rho = image.copy()
    res_min = 10
    res_max = 100
    true_rho = (true_rho - np.min(true_rho)) / (np.max(true_rho) - np.min(true_rho)) * 90 + 10  #

    XNumberPSFiTe2 = 767
    xEdgeLocationPSFiTe2 = np.linspace(fieldXStart, fieldXEnd, XNumberPSFiTe2 + 1)
    xElementLocationPSFiTe2 = 0.5 * (xEdgeLocationPSFiTe2[0:-1] + xEdgeLocationPSFiTe2[1:])
    ZNumberPSFiTe2 = 251
    zEdgeLocationPSFiTe2 = np.linspace(0, domainDepth, ZNumberPSFiTe2 + 1)
    zElementLocationPSFiTe2 = 0.5 * (zEdgeLocationPSFiTe2[0:-1] + zEdgeLocationPSFiTe2[1:])
    [xPSFiTe2, yPSFiTe2] = np.meshgrid(xElementLocationPSFiTe2, -zElementLocationPSFiTe2)
    true_rho = h_SsM.interp2_nearest(xPSFiTe2, yPSFiTe2,
                                     np.reshape(true_rho, (ZNumberPSFiTe2, XNumberPSFiTe2), order='f'), xPSFiTe,
                                     yPSFiTe)
    true_rho = np.reshape(true_rho, -1, order='f')

    h_SsM.Plot2DImage(fieldXEnd, domainDepth, xElementLocationPSFiTe, zElementLocationPSFiTe, true_rho,
                     'mt', [10**bar_1, 10**bar_2], 0, 1, './GN_inversion/' + UID,
                     '/Truth_' + 'Resistivity Model.png', rangex=[0,9.2], rangez=[-3,0],if_get_exp = 0,
                      if_jet='jet')

    # true_rho_cat = np.reshape(true_rho_cat, -1, order='f')
    # N_model_based = 20
    # code_2_pixel = np.zeros((ZNumberPSFiTe * XNumberPSFiTe, N_model_based))  # 映射矩阵
    # for pp in range(ZNumberPSFiTe * XNumberPSFiTe):
    #     num = int(true_rho_cat[pp])
    #     code_2_pixel[pp, num] = 1

    true_rho = np.log10(true_rho)

    [ia_temp, ja, value, Ub, Area, index1, Z] = MT.MT2SparseEquationSetUp_zhhy(xElementLocationMT, zElementLocationMT)

    true_rho = np.reshape(true_rho, (224, 224), order='f')

    # savemat("./Marmousi_224224.mat", {'model':true_rho})
    # pa['d'] = 3
    Nox_map_np_spar = np.zeros((ZNumberPSFiTe, XNumberPSFiTe, N_seism_corr))   # 

    ave_map = np.zeros((ZNumberPSFiTe, XNumberPSFiTe))
    for pp in range(N_seism_corr):
        No1_map_rec = get_Allpix_2_Parpix(true_rho_cat1, diff_atts[pp])
        Nox_map_np_spar[:,:,pp] = No1_map_rec
        ave_map+=No1_map_rec

    for pp in range(N_seism_corr):
        Nox_map_np_spar[:,:,pp] = Nox_map_np_spar[:,:,pp]/ave_map

    h_SsM.Plot2DImage(fieldXEnd, domainDepth, xElementLocationPSFiTe, zElementLocationPSFiTe, Nox_map_np_spar[:,:,0],
                      'mt', [0, 1], 0, 1, './GN_inversion/' + UID,
                      '/label1.png', rangex=[0, 9.2], rangez=[-3, 0], if_get_exp=0,
                      if_jet='jet')

    ###### 这是直接由速度场算出来的叠后数据
    seismic_image = np.load("post-stack_224224.npy")

    seismic_image = np.reshape(seismic_image, (ZNumberPSFiTe, XNumberPSFiTe), order='f')
    # seismic_image = seismic_image[:, ::-1]
    seismic_image = seismic_image[::-1, :]
    #####
    seismic_image = seismic_image / (np.max(abs(seismic_image)))

    neural_transfer_model = h_SsM.get_model(style_layers, layer_num, 1, ZNumberPSFiTe, XNumberPSFiTe)

    seismic_image = np.reshape(seismic_image, (ZNumberPSFiTe, XNumberPSFiTe), order='f')

    seismic_image[seismic_image > rsse] = rsse
    seismic_image[seismic_image < -rsse] = -rsse
    seismic_image1 = seismic_image / rsse
    seismic_image_tf = tf.constant(seismic_image1, dtype=tf.float64)
    image1 = tf.reshape(seismic_image_tf, (1, ZNumberPSFiTe, XNumberPSFiTe, 1))*ssit

    x_init_value = np.zeros((1, ZNumberPSFiTe, XNumberPSFiTe, 1))

    x_init_value[0, :, :, 0] = seismic_image1*ssit   ##

    x_init_value = np.reshape(x_init_value, -1, order='f')

    x_var1 = tf.constant(x_init_value, dtype=tf.float64)  #

    x_init_value1 = np.zeros((1, ZNumberMT, XNumberMT, 1)) + 1.5

    x_init_value1 = np.reshape(x_init_value1, -1, order='f')

    x_base = tf.constant(x_init_value1, dtype=tf.float64)

    x_var = tf.Variable(tf.concat([x_base, x_var1], axis=0), dtype=tf.float64, trainable=True)

    if(use_MTo_as_init==1):
        ld = loadmat('./GN_inversion/sy_Mm_pb_aV_0.0003_aH_0.0003_aS_[0.0]_Ws-[0.01, 3e-08]_fre-2.5-3.5-22_bc-[4]_g-e-x_uMai-1_rsse(0.06)_xqiw_regu2_sof-lab-25_230811改/bestModel.mat')
        init = np.squeeze(ld['model'])
        x_var = tf.Variable(init*MTit, dtype=tf.float64, trainable=True)

    print('Generate MT models and Observed data...')


    h_SsM.Plot2DImage(fieldXEnd, domainDepth, xElementLocationPSFiTe, zElementLocationPSFiTe, seismic_image1,
                      'mt', [-1, 1], 0, 1, './GN_inversion/' + UID,
                      '/Truth_' + 'Seismic Image_test.png', rangex=[0, 9.2], rangez=[-3, 0], if_get_exp=0,
                      if_jet='seismic')

    frequencyMT_tf = tf.constant(frequencyMT)

    true_rho = tf.constant(true_rho)
    true_rho = tf.cast(true_rho, dtype=tf.float64)

    ### 
    intp2d = rbs(zElementLocationPSFiTe, xElementLocationPSFiTe, true_rho,
                 [min(min(zElementLocationMT), min(zElementLocationPSFiTe)) - 5,
                  max(max(zElementLocationMT), max(zElementLocationPSFiTe)) + 5,
                  min(min(xElementLocationMT), min(xElementLocationPSFiTe)) - 5,
                  max(max(xElementLocationMT), max(xElementLocationPSFiTe)) + 5])
    true_rho_coar = intp2d(zElementLocationMT, xElementLocationMT)
    true_rho_coar = tf.reshape(tf.transpose(true_rho_coar), (-1,1))

    ## 
    # true_rho_res_f = tf.reshape(tf.transpose(true_rho), -1)
    # true_rho_coar = tf.linalg.matmul(fine_2_coar, tf.reshape(true_rho_res_f, (-1,1)))

    '''intp to coarse'''
    MT2DFWD2_packet = {'freq': frequencyMT, 'Field_rho': 0, 'Rx': RxMT, 'Field_grid_x': xElementLocationMT,
                       'Field_grid_z': zElementLocationMT, 'X_number': XNumberMT, 'Z_number': ZNumberMT,
                       'Rx_index': rxIndexMT}
    MT2DFWD2_packet['ia_temp'] = ia_temp
    MT2DFWD2_packet['ja'] = ja
    MT2DFWD2_packet['value'] = value
    MT2DFWD2_packet['Ub'] = Ub
    MT2DFWD2_packet['Area'] = Area
    MT2DFWD2_packet['index1'] = index1
    MT2DFWD2_packet['Z'] = Z

    tm1 = time.time()
    MT2DFWD2_packet['Field_rho'] = K.eval(true_rho_coar)
    if SELECT_AS_INVERSIONMODE:
        MT2DFWD2_back = pool.map(functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_packet), range(len(frequencyMT)))
    else:
        MT2DFWD2_back = []
        for ii in range(len(frequencyMT)):
            bci = MT.MT2DFWD2_zhhy(MT2DFWD2_packet, ii)
            MT2DFWD2_back.append(bci)

    EFieldVectorf = np.zeros((XNumberMT * ZNumberMT, len(frequencyMT)), dtype='complex')
    EobsVector = np.zeros(len(rxIndexMT) * len(frequencyMT), dtype='complex')
    HobsVector = np.zeros(len(rxIndexMT) * len(frequencyMT), dtype='complex')
    d_ground = np.zeros(len(rxIndexMT) * 2 * len(frequencyMT))

    for i in range(len(frequencyMT)):
        d_ground[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][:len(rxIndexMT)]
        d_ground[len(rxIndexMT) * len(frequencyMT) + i * len(rxIndexMT):len(rxIndexMT) * len(frequencyMT) + (i + 1) * len(
            rxIndexMT)] = MT2DFWD2_back[i]['data_f'][len(rxIndexMT):]
        EobsVector[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['Eobs_in']
        HobsVector[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['Hobs_in']
        EFieldVectorf[:, i] = MT2DFWD2_back[i]['EFieldVector_in']

    d_ground = tf.constant(d_ground, dtype=tf.float64)
    Logrho_f_noisea = 0.05 * tf.math.reduce_std(d_ground[:rxNumberMT * freqNumberMT]) * tf.constant(
        np.random.randn(rxNumberMT * freqNumberMT), dtype=tf.float64)
    Logrho_f_noiseb = 0.05 * tf.math.reduce_std(d_ground[rxNumberMT * freqNumberMT:]) * tf.constant(
        np.random.randn(rxNumberMT * freqNumberMT), dtype=tf.float64)
    noiss = tf.concat((Logrho_f_noisea, Logrho_f_noiseb), axis=0)
    d_ground = d_ground + noiss

    savemat('./GN_inversion/' + UID + '/Ground_truth_data.mat', {'frequencyMT':frequencyMT, 'rxLocation':xEdgeLocationMT[rxIndexMT], 'd_gnd': K.eval(d_ground)})

    if (use_MTo_as_init == 2):   ### 视电阻率作为初始模型
        fre_2_dep = np.linspace(0, zEdgeLocationMT[-1], freqNumberMT + 1)

        fre_2_dep = fre_2_dep[:-1]/2+fre_2_dep[1:]/2

        # [xsk, zsk] = np.meshgrid(xEdgeLocationMT[rxIndexMT], -fre_2_dep)

        d_ground_dg = d_ground[:rxNumberMT*freqNumberMT]

        d_ground_dg = K.eval(tf.reshape(d_ground_dg, (freqNumberMT, rxNumberMT)))

        d_ground_dg = np.flip(d_ground_dg, axis=0)

        fre_x = xElementLocationMT[rxIndexMT]
        fre_x[0] = xElementLocationMT[0]
        fre_x[-1] = xElementLocationMT[-1]
        intp2d = rbs(fre_2_dep, fre_x, d_ground_dg,
                     [min(min(zElementLocationMT), min(fre_2_dep)) - 5,
                      max(max(zElementLocationMT), max(fre_2_dep)) + 5,
                      min(min(xElementLocationMT), min(fre_x)) - 5,
                      max(max(xElementLocationMT), max(fre_x)) + 5])
        x_init_value1 = intp2d(zElementLocationMT, xElementLocationMT)

        x_init_value1 = np.reshape(x_init_value1, -1, order='f')

        x_base = tf.constant((x_init_value1)*MTit, dtype=tf.float64)

        x_var = tf.Variable(tf.concat([x_base, x_var1], axis=0), dtype=tf.float64, trainable=True)

    x_var_init, ge1, gd3 = get_property_from_code_pixel(x_var, ZNumberMT * XNumberMT)

    savemat('./GN_inversion/' + UID + '/Initial resistivity.mat',
            {"initial_model": ge1, "initial_total":x_var_init})

    h_SsM.Plot2DImage(fieldXEnd, domainDepth, xElementLocationPSFiTe, zElementLocationPSFiTe, x_var_init,
                      'mt', [bar_1, bar_2], 0, 1, './GN_inversion/' + UID,
                      '/Initial_' + 'Resistivity Model.png', rangex=[0, 9.2], rangez=[-3, 0], if_get_exp=1,
                      if_jet='jet')

    costx = tf.norm(noiss, 2) / tf.norm(d_ground, 2)
    print("Ground truth Data Misfit = {}".format(costx))
    # DR = loadmat('MarmousiMT_0228.mat')
    # d_ground = np.squeeze(DR['with_noise'])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    [xss1, zss1] = np.meshgrid(xEdgeLocationMT[rxIndexMT], np.linspace(fre1, 3.5, freqNumberMT+1))
    plt.pcolor(xss1, zss1,
               np.reshape(d_ground[:rxNumberMT * freqNumberMT], (freqNumberMT, rxNumberMT), order='c'),
               cmap=plt.get_cmap('jet'))
    cbar = plt.colorbar()
    plt.clim(1, 2)
    plt.tight_layout()
    plt.savefig('./GN_inversion/' + UID + '/Data_withnoise.png')
    plt.close()

    x_var_tmp = np.zeros(ZNumberMT * XNumberMT)
    N_front = ZNumberMT * XNumberMT
    N_back = ZNumberPSFiTe * XNumberPSFiTe
    nm = 0

    inp = tf.keras.layers.Input(shape=(N_front + N_back))
    # in1 = tf.keras.layers.Input(shape=(2*rxNumberMT))
    # in2 = tf.keras.layers.Input(shape=(2))
    cust = CustomLayer()(inp)  # no parameters in custom layer

    MT_forward_loss = tf.keras.models.Model(inputs=inp, outputs=cust)

    # lr_fn = tf.optimizers.schedules.PolynomialDecay(learning_rate1, MaxN, learning_rate2, 2)
    lr_fn = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate1, decay_steps=MaxN,
                                                    decay_rate=learning_rate2/learning_rate1)

    optimizer = tf.optimizers.Adam(learning_rate=lr_fn)

    costs = []
    cost_MT = []
    cost_Seis = []
    cost_Mater = []
    cost_norm = []
    min_MT_dm = {'ite':0, 'dm':1, 'model':0}
    x_qi_weight = np.ones(ZNumberPSFiTe*XNumberPSFiTe)
    x_var_res_f = 0

    '''with adjoint forward problem'''
    jacob_xi_2_yantuo = np.zeros((ZNumberMT * XNumberMT, ZNumberPSFiTe * XNumberPSFiTe))
    for pp in range(ZNumberMT):
        for qq in range(XNumberMT):
            jacob_xi_2_yantuo[(qq) * ZNumberMT + pp, :] = fine_2_coar[(qq) * ZNumberMT + pp, :]

    # jacob_xi_2_yantuo[jacob_xi_2_yantuo>0] = 1
    jacob_xi_2_yantuo = np.transpose(jacob_xi_2_yantuo)  ##

    for hf in range(MaxN):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_var)
            tm1 = time.time()

            x_var_hou = x_var[ZNumberMT * XNumberMT:]
            x_var_qian = x_var[:ZNumberMT * XNumberMT]

            ### 
            # x_var_4D = get_property_from_code_pixel(x_var, ZNumberMT * XNumberMT, 'tf')

            x_var_2D = tf.transpose(tf.reshape(x_var_hou, (XNumberPSFiTe, ZNumberPSFiTe)))
            x_var_4D = tf.expand_dims(x_var_2D, axis=0)
            x_var_4D = tf.expand_dims(x_var_4D, axis=3)

            
            x_var_tmp1 = tf.linalg.matmul(fine_2_coar, tf.reshape(x_var_hou, (-1, 1)))

            xvt_mat = tf.transpose(tf.reshape(x_var_tmp1, (XNumberMT, ZNumberMT)))
            xvq_mat = tf.transpose(tf.reshape(x_var_qian, (XNumberMT, ZNumberMT)))
            xvq_mat = xvq_mat + xvt_mat
            x_var_tmp = tf.reshape(tf.transpose(xvq_mat), (-1, 1))
            x_var_tmp = K.eval(x_var_tmp)

            x_var1 = tf.expand_dims(x_var, axis=0)

            costMT = MT_forward_loss(x_var1)

            tm2 = time.time()

            x_var_mat = tf.transpose(tf.reshape(x_var_qian, (XNumberMT, ZNumberMT)))

            A2 = xvq_mat[1:,:] - xvq_mat[:-1, :]   
            B2 = xvq_mat[:, 1:] - xvq_mat[:, :-1]

            if l1nm==0:
                h_constraint = get_pix_std_constraint(x_qi_weight * x_var_hou, Weis, ZNumberMT * XNumberMT)
            elif l1nm==1:
                h_constraint = Weis[1]*tf.norm(x_var_hou, 1)
            else:
                h_constraint = Weis[1]*tf.square(tf.norm(x_var_hou, 2))

            style_constraint = get_hybrid_style_constraint(image1, x_var_4D)

            if sel_norm == 'l1':
                norm_constraint = alphaVerMT * K.eval(tf.sqrt(costMT)) * tf.norm(A2, 1) + alphaHorMT * K.eval(tf.sqrt(costMT)) * tf.norm(B2, 1)
            if sel_norm == 'l2':
                norm_constraint = alphaVerMT * K.eval(tf.sqrt(costMT)) * tf.norm(A2, 2)**2 + alphaHorMT * K.eval(tf.sqrt(costMT)) * tf.norm(B2, 2)**2
            ###### PCDZ！
            costMisfitMT = norm_constraint + costMT + style_constraint + h_constraint
            #####
            cost1 = costMT
            cost2 = style_constraint
            cost3 = h_constraint
            cost4 = norm_constraint
                           # + alphaHorMT * tf.norm(hG_tf*tf.reshape(x_var, (-1,1)), 2)**2 + alphaVerMT * tf.norm(vG_tf*tf.reshape(x_var, (-1,1)), 2)**2
            costs.append(costMisfitMT)
            cost_MT.append(cost1)
            cost_Seis.append(cost2)
            cost_Mater.append(cost3)
            cost_norm.append(cost4)

            if(np.sqrt(K.eval(cost1))<min_MT_dm['dm']):
                min_MT_dm['dm'] = np.sqrt(K.eval(cost1))
                min_MT_dm['ite'] = hf
                min_MT_dm['model'] = K.eval(x_var)
                x_var_res_f, x_qi, x_ho = get_property_from_code_pixel(x_var, ZNumberMT * XNumberMT)
                x_tot_np = np.reshape(K.eval(x_var_res_f), -1, order='f')
                x_bas_np = np.reshape(K.eval(x_qi), -1, order='f')
                x_det_np = np.reshape(K.eval(x_ho), -1, order='f')
                savemat('./GN_inversion/' + UID + '/bestModel.mat', {'dm': np.sqrt(K.eval(cost1)), 'hf': hf, 'model': K.eval(x_var),
                            "total_model": x_tot_np, "base_model": x_bas_np, "detail_model": x_det_np})

            print("Ite={}, Total_cost = {:3e}, MTDM = {:3e}, cost_MT = {:3e}, cost_seis = {:3e}, cost_Mater = {:3e}, cost_norm = {:3e}, best cost MT={:3e}, best ite={}".
                  format(hf, costMisfitMT, tf.sqrt(cost1), cost1, cost2, cost3, cost4, min_MT_dm['dm'], min_MT_dm['ite']))  # (np.squeeze(grad_np)).max()

        gradients = tape.gradient(costMisfitMT, x_var)
        grad_np = K.eval(gradients)
        grad_np[grad_np != grad_np] = (np.random.rand() - 0.5) * 1E-9  # 
        nm = np.squeeze(grad_np == 0)
        nber = np.sum(nm)
        gradients = tf.convert_to_tensor(grad_np)

        if(hf < use_MTo_bef_x):
            gradients = tf.concat((gradients[:ZNumberMT*XNumberMT], tf.constant(np.zeros(ZNumberPSFiTe*XNumberPSFiTe), dtype=tf.float64)), axis=0)

        train = optimizer.apply_gradients([(gradients, x_var)])

        x_var_res_f, x_qi, x_ho = get_property_from_code_pixel(x_var, ZNumberMT*XNumberMT)

        if (np.mod(hf, 10) == 0 or hf < 10):  #

            MT2DFWD2_packet = {'freq': frequencyMT, 'Field_rho': 0, 'Rx': RxMT, 'Field_grid_x': xElementLocationMT,
                               'Field_grid_z': zElementLocationMT, 'X_number': XNumberMT, 'Z_number': ZNumberMT,
                               'Rx_index': rxIndexMT}
            MT2DFWD2_packet['ia_temp'] = ia_temp
            MT2DFWD2_packet['ja'] = ja
            MT2DFWD2_packet['value'] = value
            MT2DFWD2_packet['Ub'] = Ub
            MT2DFWD2_packet['Area'] = Area
            MT2DFWD2_packet['index1'] = index1
            MT2DFWD2_packet['Z'] = Z
            MT2DFWD2_packet['Field_rho'] = x_var_tmp
            if SELECT_AS_INVERSIONMODE:
                MT2DFWD2_back = pool.map(functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_packet), range(len(frequencyMT)))
            else:
                MT2DFWD2_back = []
                for ii in range(len(frequencyMT)):
                    bci = MT.MT2DFWD2_zhhy(MT2DFWD2_packet, ii)
                    MT2DFWD2_back.append(bci)
            d_obs = np.zeros(len(rxIndexMT) * 2 * len(frequencyMT))
            for i in range(len(frequencyMT)):
                d_obs[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][:len(rxIndexMT)]
                d_obs[len(rxIndexMT) * len(frequencyMT) + i * len(rxIndexMT):len(rxIndexMT) * len(frequencyMT) + (i + 1) * len(
                    rxIndexMT)] = MT2DFWD2_back[i]['data_f'][len(rxIndexMT):]

            d_obs = tf.constant(d_obs, dtype=tf.float64)
            costx = tf.norm(d_obs - d_ground, 2) / tf.norm(d_ground, 2)
            print("Ite={}, Data Misfit = {}".format(hf, costx))

            savemat('./GN_inversion/' + UID + '/Update_Data_' + str(hf) + '.mat', {'d_obs': K.eval(d_obs)})

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            [xss1, zss1] = np.meshgrid(xEdgeLocationMT[rxIndexMT], np.linspace(fre1, 3.5, freqNumberMT + 1))
            plt.pcolor(xss1, zss1,
                       np.reshape(d_obs[:rxNumberMT * freqNumberMT], (freqNumberMT, rxNumberMT), order='c'),
                       cmap=plt.get_cmap('jet'))
            cbar = plt.colorbar()
            plt.clim(1, 2)
            plt.tight_layout()
            plt.savefig('./GN_inversion/' + UID + '/Updated_data_{}.png'.format(hf))
            plt.close()

            h_SsM.Plot2DImage(fieldXEnd, domainDepth, xElementLocationPSFiTe, zElementLocationPSFiTe, x_var_res_f,
                              'mt', [bar_1, bar_2], 0, 1, './GN_inversion/' + UID,
                              '/Update_' + 'Image_' + str(hf) + '.png', rangex=[0, 9.2], rangez=[-3, 0],
                              if_get_exp=1, if_jet='jet')
            h_SsM.Plot2DImage(fieldXEnd, domainDepth, xElementLocationPSFiTe, zElementLocationPSFiTe, x_qi,
                              'mt', [bar_1, bar_2], 0, 1, './GN_inversion/' + UID,
                              '/Update_' + 'Image_' + str(hf) + '_blur.png', rangex=[0, 9.2], rangez=[-3, 0],
                              if_get_exp=1, if_jet='jet')
            h_SsM.Plot2DImage(fieldXEnd, domainDepth, xElementLocationPSFiTe, zElementLocationPSFiTe, x_ho,
                              'mt', [-1, 1], 0, 1, './GN_inversion/' + UID,
                              '/Update_' + 'Image_' + str(hf) + '_deta.png', rangex=[0, 9.2], rangez=[-3, 0],
                              if_get_exp=1, if_jet='jet')

        if (np.mod(hf, 20) == 0):
            kuka = np.zeros(6)
            x_tot_np = np.reshape(K.eval(x_var_res_f), -1, order='f')
            x_bas_np = np.reshape(K.eval(x_qi), -1, order='f')
            x_det_np = np.reshape(K.eval(x_ho), -1, order='f')
            # x_gr = np.reshape(K.eval(true_rho), -1, order='f')
            x_gr = np.reshape(seismic_image, -1, order='f')

            savemat('./GN_inversion/' + UID+'/Update_' + 'Image_' + str(hf) + '.mat',
                    {"total_model": x_tot_np, "base_model": x_bas_np, "detail_model": x_det_np, "true_model": x_gr})

            xt_res = x_tot_np.copy()
            xt_res = (xt_res-np.min(xt_res))/(np.max(xt_res)-np.min(xt_res))

            xg_res = x_gr.copy()
            xg_res = (xg_res - np.min(xg_res)) / (np.max(xg_res) - np.min(xg_res))

            xb_res = x_bas_np.copy()
            xb_res = (xb_res - np.min(xb_res)) / (np.max(xb_res) - np.min(xb_res))

            x = np.linspace(1, len(costs), len(costs))
            plt.figure()
            ax1 = plt.subplot(111)
            im1 = ax1.plot(x, costs, label='Total Loss', color='blue', marker='o')
            ax1.set_xlabel('No. of iteration')
            ax1.set_ylabel('Total Loss')
            plt.legend()
            ax1.set_title('Total loss')
            plt.savefig('./GN_inversion/' + UID + "/No." + str(hf) + "_Total loss.png")
            np.savetxt('./GN_inversion/' + UID + "/No." + str(hf) + "_Total loss.txt", costs, '%.5e')

            plt.figure()
            ax1 = plt.subplot(111)
            im1 = ax1.plot(x, cost_MT, label='MT', color='blue', marker='o')
            ax1.set_xlabel('No. of iteration')
            ax1.set_ylabel('MT data misfit')
            plt.legend()
            ax1.set_title('MT data misfit')
            plt.savefig('./GN_inversion/' + UID + "/No." + str(hf) + "_MT DM.png")
            np.savetxt('./GN_inversion/' + UID + "/No." + str(hf) + "_MT DM.txt", cost_MT, '%.5e')

            plt.figure()
            ax1 = plt.subplot(111)
            im1 = ax1.plot(x, cost_Seis, label='Style', color='blue', marker='o')
            ax1.set_xlabel('No. of iteration')
            ax1.set_ylabel('Style Loss')
            plt.legend()
            ax1.set_title('Style loss')
            plt.savefig('./GN_inversion/' + UID + "/No." + str(hf) + "_Style loss.png")
            np.savetxt('./GN_inversion/' + UID + "/No." + str(hf) + "_Style loss.txt", cost_Seis, '%.5e')

            plt.figure()
            ax1 = plt.subplot(111)
            im1 = ax1.plot(x, cost_Mater, label='Material', color='blue', marker='o')
            ax1.set_xlabel('No. of iteration')
            ax1.set_ylabel('Material Loss')
            plt.legend()
            ax1.set_title('Material loss')
            plt.savefig('./GN_inversion/' + UID + "/No." + str(hf) + "_Material loss.png")
            np.savetxt('./GN_inversion/' + UID + "/No." + str(hf) + "_Material loss.txt", cost_Mater, '%.5e')

            plt.figure()
            ax1 = plt.subplot(111)
            im1 = ax1.plot(x, cost_norm, label='norm', color='blue', marker='o')
            ax1.set_xlabel('No. of iteration')
            ax1.set_ylabel('Norm Loss')
            plt.legend()
            ax1.set_title('Norm loss')
            plt.savefig('./GN_inversion/' + UID + "/No." + str(hf) + "_Norm loss.png")
            np.savetxt('./GN_inversion/' + UID + "/No." + str(hf) + "_Norm loss.txt", cost_norm, '%.5e')

    Final_var = min_MT_dm['model']

    x_var_res_f, x_qi, x_ho = get_property_from_code_pixel(Final_var, ZNumberMT * XNumberMT)
    h_SsM.Plot2DImage(fieldXEnd, domainDepth, xElementLocationPSFiTe, zElementLocationPSFiTe, x_var_res_f,
                      'mt', [bar_1, bar_2], 0, 1, './GN_inversion/' + UID,
                      '/Update_best_image.png', rangex=[0, 9.2], rangez=[-3, 0],
                      if_get_exp=1, if_jet='jet')

    tm2 = time.time()
    print("Inversion Takes {} s".format(tm2-tm1))


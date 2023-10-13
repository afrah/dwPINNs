import numpy as np

import torch

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import pandas as pd
import time
import tensorflow as tf
import pickle
import random
import h5py  # for loading matlab data
import glob
import json
from datetime import datetime
from mpl_toolkits.axes_grid1 import ImageGrid
from  matplotlib.animation import PillowWriter


import logging

import torch.optim as optim
from torch.utils.data import DataLoader

# from Common import NeuralNet
# from scipy.stats import qmc

torch.manual_seed(1234)
# from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


from matplotlib.animation import FuncAnimation
import logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
random.seed(1234)
np.random.seed(1234)


###############################
os.environ["KMP_WARNINGS"] = "FALSE" 
import matplotlib.tri as tri
import seaborn as sns
import matplotlib.animation as mpa

from utilities import *


###################################################
class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


# CREATE FIGURE AND AXIS
##############################################################
    
#########################################################################

def get_testing_dataset(path ,  part='noslipwall_02_Z0slice_2D_innerdomain'):
    data =h5py.File( path, 'r')  # load dataset from matlab
    if (part == 'all'):
        WALLall = np.transpose(data['noslipwall_02_Z0slice_2D_wall'],
                               axes=range(len(data['noslipwall_02_Z0slice_2D_wall'].shape) - 1, -1, -1)).astype(np.float32)
        WALLall = np.delete(WALLall , np.where(WALLall[:,0] == WALLall[:,0].min())[0] , 0)  
        WALLall = np.delete(WALLall , np.where(WALLall[:,2] == 0.2)[0] , 0)  


        domain = np.transpose(data['noslipwall_02_Z0slice_2D_innerdomain'],
                              axes=range(len(data['noslipwall_02_Z0slice_2D_innerdomain'].shape) - 1, -1, -1)).astype(np.float32)
        
        domain = np.delete(domain , np.where(domain[:,0] == domain[:,0].min())[0] , 0)  
        domain = np.delete(domain , np.where(domain[:,2] == 0.2)[0] , 0)  

        inlet = np.transpose(data['noslipwall_02_Z0slice_2D_inlet'], 
                             axes=range(len(data['noslipwall_02_Z0slice_2D_inlet'].shape) - 1, -1, -1)).astype(np.float32)
        
        outlet = np.transpose(data['noslipwall_02_Z0slice_2D_outlet'], 
                              axes=range(len(data['noslipwall_02_Z0slice_2D_outlet'].shape) - 1, -1, -1)).astype(np.float32)
        
        XY_c = np.vstack([domain , WALLall , inlet , outlet])
    else:
        XY_c = np.transpose(data[part], axes=range(len(data[part].shape) - 1, -1, -1)).astype(np.float32)
        if part == 'noslipwall_02_Z0slice_2D_innerdomain' or part == 'noslipwall_02_Z0slice_2D_wall':
            XY_c = np.transpose(data[part], axes=range(len(data[part].shape) - 1, -1, -1)).astype(np.float32)
            XY_c = np.delete(XY_c , np.where(XY_c[:,0] == XY_c[:,0].min())[0] , 0)  
            XY_c = np.delete(XY_c , np.where(XY_c[:,1] <= 0.2)[0] , 0)  

    
    return XY_c





# generate_sobol_sequence(low , high , n)
#############################################################################################

#############################################################################################

def get_training_dataset(path , pINLET , pOUTLET , pWALL , pDomain , pInitial , dist):
    
    data = h5py.File(path , 'r')  # load dataset from matlab
    WALL = np.transpose(data['noslipwall_02_Z0slice_2D_wall'], axes=range(len(data['noslipwall_02_Z0slice_2D_wall'].shape) - 1,-1, -1)).astype(np.float32)
    
    WALL = np.delete(WALL , np.where(WALL[:,0] == WALL[:,0].min())[0] , 0)  
    # WALL = np.delete(WALL , np.where(WALL[:,1] <= 0.2)[0] , 0)  


    domain = np.transpose(data['noslipwall_02_Z0slice_2D_innerdomain'], axes=range(len(data['noslipwall_02_Z0slice_2D_innerdomain'].shape) - 1,-1, -1)).astype(np.float32)
    
    domain = np.delete(domain , np.where(domain[:,0] == domain[:,0].min())[0] , 0)  
    # domain = np.delete(domain , np.where(domain[:,1] <= 0.2)[0] , 0)  


    # INLET = np.transpose(data['noslipwall_02_Z0slice_2D_inlet'],  axes=range(len(data['noslipwall_02_Z0slice_2D_inlet'].shape) - 1,-1, -1)).astype(np.float32)
        
    INLET = domain[np.where(domain[:,1] == domain[:,1].min())[0],:] #np.delete(INLET , np.where(INLET[:,0] == INLET[:,0].min())[0] , 0)  

    OUTLET = np.transpose(data['noslipwall_02_Z0slice_2D_outlet'], axes=range(len(data['noslipwall_02_Z0slice_2D_outlet'].shape) - 1,-1, -1)).astype(np.float32)
    
    OUTLET = np.delete(OUTLET , np.where(OUTLET[:,0] == OUTLET[:,0].min())[0] , 0)  

    total = INLET.shape[0] + OUTLET.shape[0] + domain.shape[0] +  WALL.shape[0] 
    
    np.random.seed(1234)

    # initial domain ux is 0.2 and other values are zero. FOr wall, all values are zero
    
    INITIALd = domain[np.where(domain[:,0] == domain[:,0].min())[0],:] # initial doamin corressponds to all values where t is zero

    INITIALw = WALL[np.where(WALL[:,0] == WALL[:,0].min())[0],:] # initial doamin corressponds to all values where t is zero

    INITIALi = INLET[np.where(INLET[:,0] == INLET[:,0].min())[0],:] # initial doamin corressponds to all values where t is zero

    INITIALo = OUTLET[np.where(OUTLET[:,0] == OUTLET[:,0].min())[0],:] # initial doamin corressponds to all values where t is zero

    #random selection of training data
    INITIAL = np.concatenate([INITIALd , INITIALo],0)
    
    # domain = np.concatenate([domain , WALL , INLET , OUTLET],0)

    #INLET = np.delete(INLET , idx_initi , 0)  
    #OUTLET = np.delete(OUTLET , idx_inito  , 0)  
    #domain = np.delete(domain , idx_initd , 0)  
    #WALL = np.delete(WALL , idx_initw , 0)  
    
    if dist == "Sobol":
        idxi = generate_sobol_sequence(0 , INLET.shape[0] ,  int(INLET.shape[0] * pINLET)) 
        INLET = INLET[idxi, :]
        idxi = generate_sobol_sequence(0 , OUTLET.shape[0] ,  int(OUTLET.shape[0] * pOUTLET)) 
        OUTLET = OUTLET[idxi, :]
        idxi = generate_sobol_sequence(0 , INITIAL.shape[0] ,  int(INITIAL.shape[0] * pInitial)) 
        INITIAL = INITIAL[idxi, :]
        idxi = generate_sobol_sequence(0 , WALL.shape[0] ,  int(WALL.shape[0] * pWALL)) 
        WALL = WALL[idxi, :]
        idxi = generate_sobol_sequence(0 , domain.shape[0] ,  int(domain.shape[0] * pDomain)) 
        domain = domain[idxi, :]
    else:
        idxi = np.random.choice(INLET.shape[0], int(INLET.shape[0] * pINLET), replace=False)
        INLET = INLET[idxi, :]
        idxi = np.random.choice(OUTLET.shape[0], int(OUTLET.shape[0] * pOUTLET), replace=False)
        OUTLET = OUTLET[idxi, :]
        idxi = np.random.choice(INITIAL.shape[0], int(INITIAL.shape[0] * pInitial), replace=False)
        INITIAL = INITIAL[idxi, :]
        idxi = np.random.choice(WALL.shape[0], int(WALL.shape[0] * pWALL), replace=False)
        WALL = WALL[idxi, :]
        idxi = np.random.choice(domain.shape[0], int(domain.shape[0] * pDomain), replace=False)
        domain = domain[idxi, :]

    #########################################
    return [domain , INLET , OUTLET, WALL, INITIAL , total]
################################################################



def plot_result(model ,path ):


    ######################################
    model.save_NN()

    list_ = [ "noslipwall_02_Z0slice_2D_wall", "noslipwall_02_Z0slice_2D_innerdomain" ]
    N_data = [ 1600 , 20800 ]

    peotDic = dict((key,value) for key,value in zip(list_,N_data))

    tstep = [ 100 , 100 , ]
    tstepList = dict((key,value) for key,value in zip(list_,tstep))

    for key,value in peotDic.items():
        # print(key, value)
        test_data_inlet = get_testing_dataset(path   , part=key)
        test_data_inlet = test_data_inlet[np.argsort(test_data_inlet[:, 0])]

        [_ ,xf , _ , ufa , vfa  , pfa, u_pred , v_pred, p_pred]  = predict_result(model , test_data_inlet ,  value ,  tstep  ,  text=key , stm = False)
        peotDic[key] =  error_over_time(tstepList[key] ,value , ufa , vfa , pfa , u_pred , v_pred , p_pred )
        l1l2Erorr(key , u_pred ,v_pred , p_pred , ufa , vfa , pfa , model)

    draw_error_over_time(peotDic , model.dirname)

        ## drawing velocity profile
    xValues = [5.000e-04 , 0.2275, 9.995e-01]
    # UVelocity_profile(model.dirname , 0.2 , xValues , xf , u_pred)

    
    #def predict_result(model , test_data , N_data , tstep  , text = 'all'  , stm = False):
    part = 'noslipwall_02_Z0slice_2D_innerdomain'
    test_data_innerdomain = get_testing_dataset(path    ,  part=part)
    [tf , xf , yf , ufa , vfa  , pfa, u_pred , v_pred, p_pred]   = predict_result(model , test_data_innerdomain , N_data ,  tstep ,  text='domain' ,  stm = False)

    

    tstep =100
    N_data = 20800
    x = xf.reshape(tstep,N_data)[0,:]
    y = yf.reshape(tstep,N_data)[0,:]
    t = tf.reshape(tstep,N_data)[:,0].T
    ufa = ufa.reshape(tstep,N_data)
    vfa = vfa.reshape(tstep,N_data)
    pfa = pfa.reshape(tstep,N_data)
    u_pred = u_pred.reshape(tstep,N_data)
    v_pred = v_pred.reshape(tstep,N_data)
    p_pred = p_pred.reshape(tstep,N_data)
    error_u =  np.abs(ufa - u_pred) # (u - u_pred) / u # np.abs(u - u_pred)
    error_v =  np.abs(vfa - v_pred) # (v - v_pred) / v # np.abs(v - v_pred)
    error_p =  np.abs(pfa - p_pred) # (p - p_pred) / p # np.abs(p - p_pred)

    data = [u_pred, v_pred , p_pred , ufa, vfa , pfa , error_u ,  error_v ,error_p ]

    plot_time_profile(model.dirname , x , y , t , ufa , u_pred , '$u$')
    plot_time_profile(model.dirname , x , y , t , vfa , v_pred , '$v$')
    plot_time_profile(model.dirname , x , y , t , pfa , p_pred , '$p$')

    draw_contourf(t , x , y , data , 1.0 ,10 , model.dirname , 5 , fontsize=17.5 , labelsize=12.5 , axes_pad=1.2)

    model.weight_change_per_layer()








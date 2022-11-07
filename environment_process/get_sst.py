#!/student/home/zsq/wangliang/anaconda3/envs/tensorflow/bin/python3.7
#encoding:utf-8
import pandas as pd
from datetime import datetime
import os, glob
import numpy as np
from functools import reduce
import operator
import math
from math import pi
import csv
import matplotlib.pyplot as plt
from pandas import concat
def shift_uv(data,forcast_hour):
    cols=[]
    data_0=data['0']
    m=int(forcast_hour/3)
    n=int(forcast_hour/6)-1
    m=32
    n=23
    for i in range(m,n,-1):
        cols.append(data.shift(i)) 
    for i in range(n,-1,-1):
        cols.append(data_0.shift(i))  
    agg = concat(cols, axis=1)
    essemble=agg.dropna(axis=0,how='any')
    return essemble

def direct_uv(data,forecast_hour):
    dif_essemble=[]
    for i in range(len(data)):
        x_essemble=[]
        for j in range(0,8):
            u=data[i][((j+1)*441):((j+2)*441)]
            x_essemble.append(u)
        x_essemble=np.array(x_essemble)
        essemble=np.hstack(x_essemble)
        dif_essemble.append(essemble)
    return dif_essemble  

def get_data(path_track,path_uv):
    data_track=pd.read_csv(r'./'+path_track,sep=',',header=None,names=['date','lat','lon','ws','p','speed','direct'])
    #每个台风的索引的集合
    index_i=[]
    for i in range(len(data_track)):
        if data_track.date[i]=='66666':
            index_i.append(i)        
    data_track=data_track.drop(columns=['date'])
    names=[str(i) for i in range(441)]
    var_uv=pd.read_csv(r'./'+path_uv,sep=',',header=None,names=names)
    return data_track,index_i,var_uv

def minmax(array):
    ymax = 1
    ymin = 0
    xmax = max(map(max,array))
    xmin = min(map(min,array))
    for i in range(len(array)):
        for j in range(len(array[0])):
            array[i][j] = round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin,3)
    return array

def uv_file(path_track,path_uv,write_path,forcast_hour):
    data_track,index_i,var_uv=get_data(path_track,path_uv)
    with open(write_path, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for i in range(len(index_i)):
            m=index_i[i]+1       
            if i ==len(index_i)-1:
                n=len(data_track)
            else:
                n=index_i[i+1]
            uv=var_uv[m:n]
            dataframe_uv=pd.DataFrame(uv)
            dataframe_uv[np.isnan(dataframe_uv)] = -9999
            essemble_uv=shift_uv(dataframe_uv,forcast_hour)
            essemble_uv.replace(-9999,np.nan,inplace=True)
            uv_values=essemble_uv.values
            if uv_values.ndim==1:
                uv_values=uv_values.reshape(1,-1)
            essemble_uv=direct_uv(uv_values,72)
            essemble_uv=np.array(essemble_uv)
            uv_scaler=minmax(essemble_uv)
            uv_dataframe=pd.DataFrame(uv_scaler)
            uv_dataframe[np.isnan(uv_dataframe)]=0
            uv_ds=np.array(uv_dataframe)
            for row in uv_ds:
                writer.writerow(row)

uv_file('IBTrACS_fore72.txt','fore72_IBTrACS_sst.txt','./fore72_IBTrACS_sst.csv',72)
sst_train=pd.read_csv(r'./fore72_IBTrACS_sst.csv',header=None)
sst_train=sst_train.values
print(sst_train.shape)

print('over')

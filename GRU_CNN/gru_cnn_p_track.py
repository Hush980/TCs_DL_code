#!/student/home/zsq/wangliang/anaconda3/envs/tensorflow/bin/python3.7
#encoding:utf-8
seed_value= 325

import os
os.environ['PYTHONHASHSEED' ] =str(seed_value)
import random

import numpy as np
import pandas as pd
#  Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

# Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

import matplotlib.pyplot as plt
from tensorflow import keras
from numpy import concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, layers, callbacks,Model
from tensorflow.keras.layers import Dense, LSTM, Dropout,BatchNormalization,GRU, Bidirectional,SimpleRNN,Conv2D,Conv3D,Input,MaxPool3D,MaxPool2D,Flatten,TimeDistributed
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
from pandas import DataFrame
from pandas import concat
import math
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Optimizer
import tensorflow_addons as tfa
def getDistance(latA, lonA, latB, lonB):
    ra = 6378136.49  # radius of equator: meter
    rb = 6356755  # radius of polar: meter

    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)

    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    if radLonA == radLonB:
        x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(0.001))
    else:
        x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr) / 1000
    return distance


data=pd.read_csv(r'./fore72_IBTrACS_track.csv',header=None)
data=data.values
print(data.shape)
data_x=data[:,0:88]
data_y=data[:,88:112]

x_scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(data_x)
data_x = x_scaler.transform(data_x)
y_scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(data_y)
data_y = y_scaler.transform(data_y)
x_data=data_x.reshape((data_x.shape[0], 8, 11))
y_data=data_y.reshape((data_y.shape[0], 24))
print(x_data.shape,y_data.shape)
n1=36473
n2=int(n1*0.9)
train_X=x_data[:n2,:,:]
train_y=y_data[:n2,:]
valid_X=x_data[n2:n1,:,:]
valid_y=y_data[n2:n1,:]
test_X=x_data[n1:,:,:]
test_y=y_data[n1:,:]
data_latlon=data[:,112:114]
data_test=data_latlon[n1:]

uv_train=pd.read_csv(r'./fore72_IBTrACS_uv.csv',header=None)
uv_train=uv_train.values
print(uv_train.shape)
train_uv_data=uv_train[:n1,:]
train_uv=train_uv_data[:n2,:]
valid_uv=train_uv_data[n2:,:]
test_uv=uv_train[n1:,:]

train_X_uv=train_uv.reshape(train_uv.shape[0],8,8,21,21)
train_X_uv=train_X_uv.transpose(0,1,3,4,2)

valid_X_uv=valid_uv.reshape(valid_uv.shape[0],8,8,21,21)
valid_X_uv=valid_X_uv.transpose(0,1,3,4,2)

test_X_uv=test_uv.reshape(test_uv.shape[0],8,8,21,21)
test_X_uv=test_X_uv.transpose(0,1,3,4,2)

sst_train=pd.read_csv(r'./fore72_IBTrACS_sst.csv',header=None)
sst_train=sst_train.values
print(sst_train.shape)
train_sst_data=sst_train[:n1,:]
train_sst=train_sst_data[:n2,:]
valid_sst=train_sst_data[n2:,:]
test_sst=sst_train[n1:,:]

train_X_sst=train_sst.reshape(train_sst.shape[0],n_hours,1,21,21)
train_X_sst=train_X_sst.transpose(0,1,3,4,2)
valid_X_sst=train_sst.reshape(valid_sst.shape[0],n_hours,1,21,21)
valid_X_sst=valid_X_sst.transpose(0,1,3,4,2)
test_X_sst=test_sst.reshape(test_sst.shape[0],n_hours,1,21,21)
test_X_sst=test_X_sst.transpose(0,1,3,4,2)

p_train=pd.read_csv(r'./fore72_IBTrACS_p.csv',header=None)
p_train=p_train.values
print(p_train.shape)
train_p_data=p_train[:n1,:]
train_p=train_p_data[:n2,:]
valid_p=valid_p_data[n2:,:]
test_p=p_train[n1:,:]

train_X_p=train_p.reshape(train_p.shape[0],n_hours,3,46,81)
train_X_p=train_X_p.transpose(0,1,3,4,2)
valid_X_p=valid_p.reshape(valid_p.shape[0],n_hours,3,46,81)
valid_X_p=valid_X_p.transpose(0,1,3,4,2)
test_X_p=test_p.reshape(test_p.shape[0],n_hours,3,46,81)
test_X_p=test_X_p.transpose(0,1,3,4,2) 


def scheduler(epoch):
   #    # 每隔50个epoch，学习率减小为原来的1/10
    if (epoch-8)% 2 == 0 and epoch >7:
        lr = K.get_value(model.optimizer.lr)
        if lr>1e-7:
            K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

reduce_lr=LearningRateScheduler(scheduler)



# uv_input = Input(shape=(8,21,21,8), name='uv_input')
# uv_out = TimeDistributed(Conv2D(16, kernel_size=(9,9),strides=3,activation='relu'))(uv_input)
# uv_out =  TimeDistributed(BatchNormalization())(uv_out)
# uv_out =  TimeDistributed(MaxPool2D(pool_size=(3,3), strides=2, padding='valid'))(uv_out)
# uv=Flatten()(uv_out)
# uv=Dense(128,activation='relu')(uv)
# uv=Dense(32,activation='relu')(uv)

# sst_input = Input(shape=(8,21,21,1), dtype='float', name='sst_input')
# sst_out = TimeDistributed(Conv2D(8, kernel_size=(9,9),strides=3,activation='relu'))(sst_input)
# sst_out = TimeDistributed(BatchNormalization())(sst_out)
# sst_out = TimeDistributed(MaxPool2D(pool_size=(3,3), strides=2, padding='valid'))(sst_out)
# sst=Flatten()(sst_out)
# sst=Dense(128,activation='relu')(sst)
# sst=Dense(32,activation='relu')(sst)

p_input = Input(shape=(8,46,81,4), name='p_input')
p_out = TimeDistributed(Conv2D(16, kernel_size=(14,25),strides=4))(p_input)
p_out =  TimeDistributed(BatchNormalization())(p_out)
p_out = TimeDistributed(MaxPool2D(pool_size=(5,11), strides=2, padding='valid'))(p_out)
p=Flatten()(p_out)
p=Dense(128,activation='relu')(p)
p=Dense(32,activation='relu')(p)

gru_input = Input(shape=(8, 11), name='gru_input')
layer_1 = GRU(128,return_sequences=True,activation='relu')(gru_input)
layer_2 = GRU(32,return_sequences=False,activation='relu')(layer_1)
layer_3 = tf.concat([p,layer_2],axis=-1)
output=Dense(24,,activation='relu')(layer_3)
model = Model(inputs=[p_input,gru_input], outputs=output)
# lr_0= tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-5,decay_steps=2280,decay_rate=0.94)
# lr_1= tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=8e-5,decay_steps=2280,decay_rate=0.94)
# lr_2=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=2280,decay_rate=0.92)
# lr_3= tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=3e-4,decay_steps=2280,decay_rate=0.92)

# optimizers = [tf.keras.optimizers.Adam(learning_rate=lr_0),
# tf.keras.optimizers.Adam(learning_rate=lr_1),
# tf.keras.optimizers.Adam(learning_rate=lr_2),
# tf.keras.optimizers.Adam(learning_rate=lr_3)]


# optimizers_and_layers = [(optimizers[0], model.layers[3]), (optimizers[1], model.layers[4]),(optimizers[2], model.layers[5]),
# (optimizers[0], model.layers[16]),(optimizers[1], model.layers[17]),(optimizers[2], model.layers[18]),(optimizers[3], model.layers[19]),
# (optimizers[0], model.layers[20]),(optimizers[1], model.layers[21]),(optimizers[2], model.layers[22]),(optimizers[3], model.layers[23]),]
# optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=5000,decay_rate=0.95)
optimizer=tf.keras.optimizers.Adam(learning_rate=lr)

# adam=Adam(lr=0.003)
model.compile(loss='mean_squared_error', optimizer=lr,metrics=['accuracy', 'mae', 'mape'])
save_to = './cnn_gru_p_72.hdf5'
es=tf.keras.callbacks.ModelCheckpoint(save_to,monitor='val_loss',save_best_only=True,save_weights_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.summary()
history = model.fit([train_X_p,train_X],train_y,validation_data=([valid_X_p,valid_X],valid_y) ,epochs=100,batch_size=16, callbacks=[es,reduce_lr],verbose=2,workers=5)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.legend()
plt.show()

# make a prediction

model.load_weights('./cnn_gru_p_72.hdf5')
tf.compat.v1.enable_eager_execution()
yhat=model([test_X_p,test_X])
pre_yhat= y_scaler.inverse_transform(yhat)
act_y = y_scaler.inverse_transform(test_y)

for j in range(0, 24):
    inv_yhat = pre_yhat[:, j]
    inv_y = act_y[:,j]
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test{:} RMSE: %.3f'.format(j) % rmse)
lat0=data_test[:,0]
lon0=data_test[:,1]

dis_all_all=[]
for i in range(len(pre_yhat)):
    dis_all_ds=[]
    for j in range(12):
        lat_pred_all=pre_yhat[i,j*2]+lat0[i]
        lon_pred_all=pre_yhat[i,j*2+1]+lon0[i]
        lat_true=act_y[i,j*2]+lat0[i]
        lon_true=act_y[i,j*2+1]+lon0[i]
        dis=getDistance(lat_pred_all, lon_pred_all, lat_true, lon_true)
        dis_all_ds.append(dis)
    dis_all_all.append(dis_all_ds)
lat_0=lat0.reshape(-1,1)
lon_0=lon0.reshape(-1,1)
print(np.mean(dis_all_all,axis=0))
true=np.concatenate((act_y,lat_0,lon_0),axis=1)
import csv
def essemble_path(name,result):
    path='./'+name
#     result_T=list(map(list, zip(*result)))
    with open(path, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in result:
            writer.writerow(row)
#
essemble_path('result_cnn_gru_p_fore72.csv',pre_yhat)

#essemble_path('result_true.csv',true)

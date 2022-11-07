seed_value= 325

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED' ] =str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
import pandas as pd
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

# 5. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Visualization
import matplotlib.pyplot as plt

import numpy as np
from tensorflow import keras
from numpy import concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, layers, callbacks,Model
from tensorflow.keras.layers import Dense,LeakyReLU, LSTM, Dropout,BatchNormalization,GRU, Bidirectional,SimpleRNN,Conv2D,Conv3D,Input,MaxPool3D,MaxPool2D,Flatten,TimeDistributed
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
from pandas import DataFrame
from pandas import concat
import math
from tensorflow.keras.layers import Flatten
#from lstm_attention import Attention
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

def minmax(array):
   ymax = 1
   ymin = 0
   xmax = max(map(max,array))
   xmin = min(map(min,array))

   for i in range(len(array)):
      for j in range(len(array[0])):
         array[i][j] = round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin,3)
   return array
n_hours=8
n_features=11

data=pd.read_csv(r'/content/drive/MyDrive/ctrl_RNN/fore72_IBTrACS_track.csv',header=None)
data=data.values
print(data.shape)
data_x=data[:,0:88]
data_y=data[:,88:112]

x_scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(data_x)
data_x = x_scaler1.transform(data_x)
y_scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(data_y)
data_y = y_scaler1.transform(data_y)
x_data=data_x.reshape((data_x.shape[0], 8, 11))
y_data=data_y.reshape((data_y.shape[0], 24))
print(x_data.shape,y_data.shape)
n1=36473
data_train_x=x_data[:n1,:,:]
data_train_y=y_data[:n1,:]
data_test_x=x_data[n1:,:,:]
data_test_y=y_data[n1:,:]
data_latlon=data[:n1,112:114]
data_test=data[n1:,112:114]

n2=int(len(data_train_x)*0.9)
train_X=data_train_x[:n2,:,:]
train_y=data_train_y[:n2,:]
valid_X=data_train_x[n2:,:,:]
valid_y=data_train_y[n2:,:]


test_X=data_test_x[:,:,:]
test_y=data_test_y[:,:]
def scheduler(epoch):
   #    # 每隔50个epoch，学习率减小为原来的1/10
    if (epoch-4) % 4 == 0 and epoch >3:
        lr = K.get_value(model.optimizer.lr)
        if lr>1e-5:
            K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)

lstm_input = Input(shape=(train_X.shape[1], train_X.shape[2]), name='lstm_input')
layer_1 = LSTM(128,return_sequences=True)(lstm_input)
layer_2 = LSTM(64,return_sequences=False)(layer_1)
# layer_3=LeakyReLU(alpha=0.1)(layer_3)
layer_3=Dense(32)(layer_2)
layer_4=Dense(24)(layer_3)
model = Model(inputs=[lstm_input], outputs=layer_4)
lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-4,decay_steps=2280,decay_rate=0.98)
optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mae', 'mape'])
save_to = '/content/drive/MyDrive/ctrl_RNN/ctrl_cnn_lstm_72_track.hdf5'
es=tf.keras.callbacks.ModelCheckpoint(save_to,monitor='val_loss',save_best_only=True,save_weights_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# es = EarlyStopping(monitor='loss', patience=10)
#history = model.fit([train_X],train_y,validation_data=([valid_X],valid_y) ,epochs=200,batch_size=16,  callbacks=[es,reduce_lr],verbose=2)
#model.save_weights('./ctrl_cnn_lstm_72_attention_3_1.hdf5')
model.summary()
# plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='valid')
#plt.legend()
#plt.show()

# make a prediction

#model=tf.keras.models.load_model('./ctrl_cnn_lstm_72_attention.hdf5')
model.load_weights('/content/drive/MyDrive/ctrl_RNN/ctrl_cnn_lstm_72_track.hdf5')
tf.compat.v1.enable_eager_execution()
pre_yhat_ds=[]
for i in range(2):
    yhat=model([train_X])
    pre_yhat= y_scaler1.inverse_transform(yhat) 
    pre_yhat_ds.append(pre_yhat)
pre_yhat=np.mean(pre_yhat_ds,axis=0)
act_y = y_scaler1.inverse_transform(train_y)

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

print(np.mean(dis_all_all,axis=0))

import csv
def essemble_path(name,result):
    path='/'+name
#     result_T=list(map(list, zip(*result)))
    with open(path, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in result:
            writer.writerow(row)
#
# essemble_path('content/drive/MyDrive/ctrl_RNN/result_ctrl_cnn_lstm_fore48.csv',pre_yhat)



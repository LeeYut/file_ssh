# -*- coding: utf-8 -*-   
import numpy as np
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.models import model_from_json
import os
#import matplotlib.pyplot as plt
#返回值是一个array,是(969,)的形式		
def read_file(folder_name,tag):
    file_list = os.listdir(folder_name)
    result = np.array([])
    for file in file_list:
        df = pd.read_csv(folder_name+ '/' + file)
        data = np.array(df[tag].values)
        result = np.concatenate((result, data), axis = 0)
    return result

#scaling的对象不能是单纯一维的，所以要reshape. [1,2,3,4] -> [[1],[2],[3],[4]],成为了一列，均值就是指的这一列的，是2.5
def scaling(result):
    result = result.reshape(-1,1)
    scaler = preprocessing.StandardScaler().fit(result)
    mean = scaler.mean_
    scale = scaler.scale_
    data = scaler.transform(result)
    return mean, scale, data


#这里将merge后的数据，整理成为带有time_step的数据
def time_transform(merge, time_step, type):
    merge_array = np.array(merge)
    x, y = [],[]
    #convert merge_array to array with time_step
    for i in range(len(merge) - time_step + 1):
        x.append(merge_array[i:i+time_step])
    if(type == 'esc_up'):
        y = np.array([1,0,0,0,0,0]*len(x))
    elif(type == 'esc_down'):
        y = np.array([0,1,0,0,0,0]*len(x))
    elif(type == 'same_floor'):
        y = np.array([0,0,1,0,0,0]*len(x))
    elif(type == 'elevator_up'):
        y = np.array([0,0,0,1,0,0]*len(x))
    elif(type == 'elevator_down'):
        y = np.array([0,0,0,0,1,0]*len(x))
    else:
        y = np.array([0,0,0,0,0,1]*len(x))	

    X = np.array(x)
    #这里y要变形，这里的y就是一个纯的list，注意如果有三类数据就需要，reshape的第三维度是3
    Y = y.reshape(-1,6)
    return X, Y

def create_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences = False))
    #model.add(LSTM(100))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=30, batch_size=50, validation_split = 0.2, verbose=2)
    return model
	
#文件夹的形式是 模式类别
#             /加速度数据文件夹，  /气压数据文件夹
#               /所有加速度文件  /所有气压文件
#注意存放的名字和顺序必须对应 

baro_ele_up = read_file('elevator_up/baro', 'p')
baro_ele_down = read_file('elevator_down/baro', 'p')
baro_esc_up = read_file('escalator_up/baro', 'p')
baro_esc_down = read_file('escalator_down/baro', 'p')
baro_same_floor = read_file('same_floor/baro', 'p')

#merge后出来的结果其实是一个list套list，[[],[],[]]，并不是numpy的array
# a = merge(acc_esc_up, baro_esc_up,mag_esc_up)
# b = merge(acc_esc_down,baro_esc_down,mag_esc_down)
# c = merge(acc_ele_up,baro_ele_up,mag_ele_up)
# d = merge(acc_ele_down,baro_ele_down,mag_ele_down)
# e= merge(acc_same_floor,baro_same_floor,mag_same_floor)
# f = merge(acc_subway, baro_subway, mag_subway)

#concatenate真的很棒，普通的list也可以被串起来
# a_b_merge = np.concatenate((a,b,c,d,e,f), axis = 0)
a_b_merge = np.concatenate((baro_ele_up,baro_ele_down,baro_esc_up,baro_esc_down,baro_same_floor), axis = 0)


#以下部分用于对于全部的数据求出正确的均值和方差，用于测试组正规化数据
scaler = preprocessing.StandardScaler()
scaler.fit(a_b_merge.reshape(-1,1))

#对数据组a b进行正规化
a_norm = scaler.transform(baro_ele_up.reshape(-1,1))
b_norm = scaler.transform(baro_ele_down.reshape(-1,1))
c_norm = scaler.transform(baro_esc_up.reshape(-1,1))
d_norm = scaler.transform(baro_esc_down.reshape(-1,1))
e_norm = scaler.transform(baro_same_floor.reshape(-1,1))


#对a b分别做含有time step的变换,输入的是有多个特征的情形
x1, y1 = time_transform(a_norm, 80, 'elevator_up')
x2, y2 = time_transform(b_norm, 80, 'elevator_down')
x3, y3 = time_transform(c_norm, 80, 'esc_up')
x4, y4 = time_transform(d_norm, 80, 'esc_down')
x5, y5 = time_transform(e_norm, 80, 'same_floor')

#合并并且打乱顺序
# X = np.concatenate((x1,x2,x3,x4,x5,x6), axis = 0)
# Y = np.concatenate((y1, y2,y3,y4,y5,y6), axis = 0)

X = np.concatenate((x1,x2,x3,x4,x5), axis = 0)
Y = np.concatenate((y1, y2,y3,y4,y5), axis = 0)
X, Y = shuffle(X, Y)


#数据已经准备完成，可以送去训练模型了
model = create_model(X, Y)
#保存模型
model_json = model.to_json()
with open("model_pressure_only_without3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_pressure_only_without3.h5")
print("Saved model to disk")
print ('mean')
print (scaler.mean_)
print ('scale')
print (scaler.scale_)

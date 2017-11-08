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
from keras.layers import Dropout
import matplotlib.pyplot as plt
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

#输入是scaling之后的数据，我们这里将两个zip起来，注意zip只对[1,2,3],[4,5,6]工作得很好
def merge(data1, data2, data3):
    merge = []
    data1 = data1.reshape(1, len(data1))
    data2 = data2.reshape(1, len(data2))
    data3 = data3.reshape(1, len(data3))

    #print (data1.shape)
    z = zip(data1[0], data2[0],data3[0])
    for i in list(z):
        merge.append(list(i))
    return merge

	
# def merge(data1, data2):
    # merge = []
    # data1 = data1.reshape(1, len(data1))
    # data2 = data2.reshape(1, len(data2))

    print (data1.shape)
    # z = zip(data1[0], data2[0])
    # for i in list(z):
        # merge.append(list(i))
    # return merge
	

#这里将merge后的数据，整理成为带有time_step的数据
def time_transform(merge, time_step, type):
    merge_array = np.array(merge)
    x, y = [],[]
    #convert merge_array to array with time_step
    for i in range(len(merge) - time_step + 1):
        x.append(merge_array[i:i+time_step])
    if(type == 'subway'):
        y = np.array([1,0]*len(x))
  
    else:
        y = np.array([0,1]*len(x))	

    X = np.array(x)
    #这里y要变形，这里的y就是一个纯的list，注意如果有三类数据就需要，reshape的第三维度是3
    Y = y.reshape(-1,2)
    return X, Y

def create_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences = False))
    # model.add(LSTM(100))
    # model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences = True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(100))
    # model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=30, batch_size=100, validation_split = 0.3, verbose=2)
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
baro_subway = read_file('subway/baro', 'p')

acc_ele_up = read_file('elevator_up/acc', 'acc')
acc_ele_down = read_file('elevator_down/acc', 'acc')
acc_esc_up = read_file('escalator_up/acc', 'acc')
acc_esc_down = read_file('escalator_down/acc', 'acc')
acc_same_floor = read_file('same_floor/acc', 'acc')
acc_subway = read_file('subway/acc','acc')

mag_ele_up = read_file('elevator_up/mag', 'm')
mag_ele_down = read_file('elevator_down/mag', 'm')
mag_esc_up = read_file('escalator_up/mag', 'm')
mag_esc_down = read_file('escalator_down/mag', 'm')
mag_same_floor = read_file('same_floor/mag', 'm')
mag_subway = read_file('subway/mag','m')

#merge后出来的结果其实是一个list套list，[[],[],[]]，并不是numpy的array
a = merge(acc_esc_up, baro_esc_up,mag_esc_up)
b = merge(acc_esc_down,baro_esc_down,mag_esc_down)
c = merge(acc_ele_up,baro_ele_up,mag_ele_up)
d = merge(acc_ele_down,baro_ele_down,mag_ele_down)
e= merge(acc_same_floor,baro_same_floor,mag_same_floor)
f = merge(acc_subway, baro_subway, mag_subway)

#concave】tenate真的很棒，普通的list也可以被串起来
# a_b_merge = np.concatenate((a,b,c,d,e,f), axis = 0)
# a_b_merge = np.concatenate((a,b,f), axis = 0)

max_value = [20, 0.03, 200]
a = np.array(a)
b = np.array(b)
c = np.array(c)
d = np.array(d)
e = np.array(e)
f = np.array(f)



#以下部分用于对于全部的数据求出正确的均值和方差，用于测试组正规化数据
a[:,0:1] = (a[:,0:1]-max_value[0])/max_value[0]
a[:,1:2] = (a[:,1:2]-max_value[1])/max_value[1]
a[:,2:3] = (a[:,2:3]-max_value[2])/max_value[2]



b[:,0:1] = (b[:,0:1]-max_value[0])/max_value[0]
b[:,1:2] = (b[:,1:2]-max_value[1])/max_value[1]
b[:,2:3] = (b[:,2:3]-max_value[2])/max_value[2]

c[:,0:1] = (c[:,0:1]-max_value[0])/max_value[0]
c[:,1:2] = (c[:,1:2]-max_value[1])/max_value[1]
c[:,2:3] = (c[:,2:3]-max_value[2])/max_value[2]

d[:,0:1] = (d[:,0:1]-max_value[0])/max_value[0]
d[:,1:2] = (d[:,1:2]-max_value[1])/max_value[1]
d[:,2:3] = (d[:,2:3]-max_value[2])/max_value[2]

e[:,0:1] = (e[:,0:1]-max_value[0])/max_value[0]
e[:,1:2] = (e[:,1:2]-max_value[1])/max_value[1]
e[:,2:3] = (e[:,2:3]-max_value[2])/max_value[2]

f[:,0:1] = (f[:,0:1]-max_value[0])/max_value[0]
f[:,1:2] = (f[:,1:2]-max_value[1])/max_value[1]
f[:,2:3] = (f[:,2:3]-max_value[2])/max_value[2]


#对a b分别做含有time step的变换,输入的是有多个特征的情形
x1, y1 = time_transform(a, 80, 'esc_up')
x2, y2 = time_transform(b, 80, 'esc_down')
x3, y3 = time_transform(c, 80, 'ele_up')
x4, y4 = time_transform(d, 80, 'ele_down')
x5, y5 = time_transform(e, 80, 'same_floor')
x6, y6 = time_transform(f, 80, 'subway')

#合并并且打乱顺序
X = np.concatenate((x1,x2,x3,x4,x5,x6), axis = 0)
Y = np.concatenate((y1, y2,y3,y4,y5,y6), axis = 0)
# plt.subplot(2,1,1)
# plt.plot(X)
# plt.subplot(2,1,2)
# plt.plot(np.argmax(Y,axis = 1))
# X = np.concatenate((x1,x2,x6), axis = 0)
# Y = np.concatenate((y1, y2,y6), axis = 0)
X, Y = shuffle(X, Y)
print (X.shape)
print (Y[:50])

#数据已经准备完成，可以送去训练模型了
model = create_model(X, Y)
#保存模型
model_json = model.to_json()
with open("model_binary_1108.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_binary_1108.h5")
print("Saved model to disk")
# print ('mean')
# print (scaler.mean_)
# print ('scale')
# print (scaler.scale_)
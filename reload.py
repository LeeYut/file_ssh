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

def read_file(folder_name,tag):
    file_list = os.listdir(folder_name)
    result = np.array([])
    for file in file_list:
        df = pd.read_csv(folder_name+ '/' + file)
        data = np.array(df[tag].values)
        result = np.concatenate((result, data), axis = 0)
    return result
	
def time_transform(merge, time_step, type):
    merge_array = np.array(merge)
    x, y = [],[]
    #convert merge_array to array with time_step
    for i in range(len(merge) - time_step + 1):
        x.append(merge_array[i:i+time_step])
    if(type == 'esc_up'):
        y = np.array([1,0,0,0,0]*len(x))
    elif(type == 'esc_down'):
        y = np.array([0,1,0,0,0]*len(x))
    elif(type == 'same_floor'):
        y = np.array([0,0,1,0,0]*len(x))
    elif(type == 'elevator_up'):
        y = np.array([0,0,0,1,0]*len(x))
    else:
        y = np.array([0,0,0,0,1]*len(x))

    X = np.array(x)
    #这里y要变形，这里的y就是一个纯的list，注意如果有三类数据就需要，reshape的第三维度是3
    Y = y.reshape(-1,5)
    return X, Y
	
#输入是scaling之后的数据，我们这里将两个zip起来，注意zip只对[1,2,3],[4,5,6]工作得很好
def merge(data1, data2):
    merge = []
    data1 = data1.reshape(1, len(data1))
    data2 = data2.reshape(1, len(data2))
    print (data1.shape)
    z = zip(data1[0], data2[0])
    for i in list(z):
        merge.append(list(i))
    return merge	
	
#-----------------------测试阶段---------------------
#读入数据,平均值和标准差要从训练model的输出得到
# mean=[10.3412734, 1014.54918458]
# scale = [ 2.74341942 , 0.56282084]
# acc_test = read_file('data/walkup_test/acc', 'acc')
# baro_test = read_file('data/lift_down_test/baro', 'p')
# test = np.array(merge(acc_test, baro_test))
# test[:,0:1] = (test[:,0:1]-mean[0])/scale[0]
# test[:,1:2] = (test[:,1:2]-mean[1])/scale[1]
# x1, y1 = time_transform(test, 10, 'elevator')
# print (x1)

baro_esc_up = read_file('baro', 'p')
# baro_esc_down = read_file('escalator_down/baro', 'p')

x1, y1 = time_transform(baro_esc_up, 40, 'esc_down')
# x2, y2 = time_transform(baro_esc_down, 40, 'esc_down')

x1 = x1.reshape(len(x1), len(x1[0]), 1)
# x2 = x2.reshape(len(x2), len(x2[0]), 1)

# X = np.concatenate((x1,x2,x3,x4,x5), axis = 0)
# Y = np.concatenate((y1, y2,y3,y4,y5), axis = 0)
X = x1

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.predict(X)
# print (np.argmax(score, axis = 1))
a = np.argmax(score, axis = 1)
cnt = 0
for i in a:
    if(i == 3):
        cnt += 1
print(cnt/len(a))
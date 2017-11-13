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
from matplotlib import pyplot as plt
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
    # for i in range(len(merge) - time_step + 1):
        # x.append(merge_array[i:i+time_step])
    for i in range(int(len(merge)/time_step)):
        x.append(merge_array[i*time_step:i*time_step+time_step])
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
	
#-----------------------测试阶段---------------------
#读入数据,平均值和标准差要从训练model的输出得到

mean_scale = [-0.00017236, 0.00764564]


test = read_file('same_floor/test/baro', 'p').reshape(-1,1)

print (test)
test[:,0:1] = (test[:,0:1]-mean_scale[0])/mean_scale[1]

x1, y1 = time_transform(test, 80, 'esc_up')


json_file = open('model_pressure_only.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("model_pressure_only.h5")
print("Loaded model from disk")
 
#evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.predict(x1)
print (np.argmax(score, axis = 1))
a = np.argmax(score, axis = 1)
plt.plot(a)
plt.ylim(-1,6)
plt.show()
cnt = 0
for i in a:
    if(i == 2 ):
        cnt += 1
print(cnt/len(a))
print(score)
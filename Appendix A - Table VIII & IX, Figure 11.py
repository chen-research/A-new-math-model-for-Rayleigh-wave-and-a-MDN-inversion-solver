#This file contains codes for all results in Appendix A
#Table VII and VIII
#Figure 9
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cluster import OPTICS
from tensorflow.compat.v1 import keras
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd
from Geophysics_Package import *

#To limit TensorFlow to a specific set of GPUs
pd.set_option('display.max_columns', None)

data_path = "D:/Large_Files_For_Learning/Project_Result_Data/With_Yang_Zhang/Geophysics/"
#figure_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/Drafts/Figures/"
data3 = pd.read_csv(data_path+"layers3_kuanyueshu.csv",header=None)[0:48000] #3 layers data
data5 = pd.read_csv(data_path+"layers5_kuanyueshu.csv",header=None)[0:48000] #5 layers data
data9 = pd.read_csv(data_path+"layers9_kuanyueshu.csv",header=None) #9 layers data

data3 = data3.sample(frac=1,axis=0,replace=False,random_state=48)  #shuffle the data
data5 = data5.sample(frac=1,axis=0,replace=False,random_state=48)  #shuffle the data
data9 = data9.sample(frac=1,axis=0,replace=False,random_state=48)  #shuffle the data

data3 = data3.reset_index(drop=True)   #reset the index
data5 = data5.reset_index(drop=True)   #reset the index
data9 = data9.reset_index(drop=True)   #reset the index

#pd.set_option('display.max_columns', None)
print('Data3 shape before dropping duplicates',data3.shape, 'after duplicates are dropped', data3.drop_duplicates().shape)
print('Data5 shape before dropping duplicates',data5.shape, 'after duplicates are dropped', data5.drop_duplicates().shape)
print('Data9 shape before dropping duplicates',data9.shape, 'after duplicates are dropped', data9.drop_duplicates().shape)



######################
#Appendix A 
#Table VIII -- 1st row (3-layer)
#Build and train the 3layer FNN with the optimal hyper-parameters, bw_model3: C->vs
######################

# --- Data for the 3 layer bw_model: C to vs
output_cols = list(range(2,5)) #list of column names for outputs, Vs
input_cols = list(range(114-50,114)) #C
inputs = data3[input_cols]
outputs = data3[output_cols]

#convert the data to np.array
inputs = np.array(inputs)
outputs = np.array(outputs)

sample_size = len(inputs)
tr_size = int(sample_size*9/10)
test_size = int(sample_size*1/10)

train_test_size = tr_size+test_size
train_x, test_x, train_y, test_y = train_test_split(inputs[0:train_test_size], outputs[0:train_test_size], test_size=test_size/(tr_size+test_size), random_state=48)

param3_m = np.mean(train_x,axis=0)
param3_sd = np.std(train_x,axis=0)

##### --- Train bw_model3
bw_model3, train_R2, test_R2 = train_model(train_x, train_y, test_x, test_y, 
                                        loss_func= 'L2', #The loss function
                                        activ= 'tanh', #The activation function
                                        hid_layers= (150,150,150,100,100), #(1st hidden layer nodes, 2nd hidden layer nodes)
                                        bias_regu_cosnt= 0.001, #The regularization coeff. for bias terms
                                        w_regu_const= 0.001, #The regularization coeff. for weights
                                        epoch_num= 500, #The number of maximum epochs that will performed. 
                                        early_stop_patience= 10, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                                        show_process= 0 #If equals 1, show the training process, if equals 0, do not show. 
                                        )
print("Test_R2:",round(test_R2,3), "Train_R2:",round(train_R2,3))   



######################
#Appendix A 
#Table VIII -- 2nd row (5-layer)
#Builds and trains 5layer FNN with the optimal hyper-parameters, bw_model5: C->vs
######################

##### --- Data for the 5 layer bw_model: C to vs
output_cols = list(range(4,9)) #list of column names for outputs, Vs
input_cols = list(range(124-50,124)) #C
inputs = data5[input_cols]
outputs = data5[output_cols]

#convert the data to np.array
inputs = np.array(inputs)
outputs = np.array(outputs)

sample_size = len(inputs)
tr_size = int(sample_size*9/10)
test_size = int(sample_size*1/10)

train_test_size = test_size+tr_size
train_x, test_x, train_y, test_y = train_test_split(inputs[0:train_test_size], outputs[0:train_test_size], test_size=test_size/(tr_size+test_size), random_state=48)

param5_m = np.mean(train_x,axis=0)
param5_sd = np.std(train_x,axis=0)

##### --- Train bw_model5
bw_model5, train_R2, test_R2 = train_model(train_x, train_y, test_x, test_y, 
                                        loss_func= 'L2', #The loss function
                                        activ= 'tanh', #The activation function
                                        hid_layers= (150,150,150,100,100), #(1st hidden layer nodes, 2nd hidden layer nodes)
                                        bias_regu_cosnt= 0.001, #The regularization coeff. for bias terms
                                        w_regu_const= 0.001, #The regularization coeff. for weights
                                        epoch_num= 500, #The number of maximum epochs that will performed. 
                                        early_stop_patience= 10, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                                        show_process= 0 #If equals 1, show the training process, if equals 0, do not show. 
                                        )
print("Test_R2:",round(test_R2,3), "Train_R2:",round(train_R2,3))   



######################
#Appendix A 
#Table VIII -- 3rd row (9-layer)
#Table IX
#Builds and trains 9layer FNN with the optimal hyper-parameters, bw_model9: C->vs
######################

##### --- Data for the 9 layer bw_model: C to vs
output_cols = list(range(8,17)) #list of column names for outputs, Vs
input_cols = list(range(144-50,144)) #C
inputs = data9[input_cols]
outputs = data9[output_cols]

#convert the data to np.array
inputs = np.array(inputs)
outputs = np.array(outputs)

sample_size = len(inputs)
tr_size = int(sample_size*9/10)
test_size = int(sample_size*1/10)

train_test_size = test_size+tr_size
train_x, test_x, train_y, test_y = train_test_split(inputs[0:train_test_size], outputs[0:train_test_size], test_size=test_size/(tr_size+test_size), random_state=48)

param9_m = np.mean(train_x,axis=0)
param9_sd = np.std(train_x,axis=0)

##### --- Train bw_model9
bw_model9, train_R2, test_R2 = train_model(train_x, train_y, test_x, test_y, 
                                        loss_func= 'L2', #The loss function
                                        activ= 'tanh', #The activation function
                                        hid_layers= (400,300,300,300,300), #(1st hidden layer nodes, 2nd hidden layer nodes)
                                        bias_regu_cosnt= 0.001, #The regularization coeff. for bias terms
                                        w_regu_const= 0.001, #The regularization coeff. for weights
                                        epoch_num= 100, #The number of maximum epochs that will performed. 
                                        early_stop_patience= 10, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                                        show_process= 0 #If equals 1, show the training process, if equals 0, do not show. 
                                        )
print("Test_R2:",round(test_R2,3), "Train_R2:",round(train_R2,3))   


#Appendix A 
#Figure 11 -- Illustrate the One-to-Many Phenomenon in the dataset
cluster_range = 10000
clustering = OPTICS(min_samples=2,max_eps=0.05,metric='euclidean').fit(inputs[0:cluster_range])
partial_data = data9[input_cols+output_cols][0:cluster_range].copy()
partial_data['labels'] = clustering.labels_     
print('Number of groups:',len(set(clustering.labels_)))
print('number of grouped data', len(partial_data[partial_data.labels>=0]))
print('number of un-grouped data', len(partial_data[partial_data.labels==-1]))
partial_data.sort_values(by=['labels'],inplace=True,ignore_index=False)      
i, j = partial_data.index[-2:] #Inputs of the ith and jth data samples are close

#i = 8513
#j = 9577
f, ax =plt.subplots(1,2,figsize=(12,4)) 
x_coord = range(inputs.shape[1])
ax[1].scatter(x_coord,inputs[i],color='b',label="Sample_1 "+r"$\bf{y}$")
ax[1].scatter(x_coord,inputs[j],color='r',label="Sample_2 "+r"$\bf{y}$")
ax[1].set_title(r"$\bf{y}$" + " values of two samples",fontsize=16)
ax[1].set_xlabel('Index',fontsize=15)
ax[1].tick_params(labelsize=14)
ax[1].legend(loc='upper left',fontsize=15)

#x_coord = range(outputs.shape[1])
x_coord = range(1,10)
ax[0].scatter(x_coord,outputs[i,0:9],color='b',label="Sample_1 " + r"$\bf{x}$")
ax[0].scatter(x_coord,outputs[j,0:9],color='r',label="Sample_2 " + r"$\bf{x}$")
ax[0].set_title(r"$\bf{x}$" + " of two samples",fontsize=16)
ax[0].set_xlabel('index',fontsize=15)
ax[0].tick_params(labelsize=14)
ax[0].legend(loc='upper left',fontsize=15)
#plt.savefig(figure_path+'Fig8.eps',bbox_inches="tight")
plt.show()

#Appendix B - Table X
#This file investigates the impact of K's value on model's validation performances

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cluster import OPTICS
from tensorflow.compat.v1 import keras
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd
from Geophysics_Package import *


np.random.seed(1) #Set random seed for numpy random generator
#To limit TensorFlow to a specific set of GPUs
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data_path = "D:/Large_Files_For_Learning/Project_Result_Data/With_Yang_Zhang/Geophysics/"
model_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/python_codes/Clean_Version/"
#figure_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/Drafts/Figures/"
data9 = pd.read_csv(data_path+"layers9_kuanyueshu.csv",header=None) #9 layers data
data9 = data9.sample(frac=1,axis=0,replace=False,random_state=1)  #shuffle the data
data9 = data9.reset_index(drop=True)   #reset the index

#pd.set_option('display.max_columns', None)
print('Data9 shape before dropping duplicates',data9.shape, 'after duplicates are dropped', data9.drop_duplicates().shape)


######################
#Data for training fw-FNN: i.e., the 9layer forward model fw_model: vs -> C
#input_cols = [str(i) for i in range(8,17)] #list of column names for outputs
#output_cols = [str(i) for i in range(144-50,144)]
######################

input_cols = list(range(8,17)) #list of column names for outputs
output_cols = list(range(144-50,144))
inputs = data9[input_cols]
outputs = data9[output_cols]

#convert the data to np.array
inputs = np.array(inputs)
outputs = np.array(outputs)

sample_size = len(inputs)
tr_size = int(sample_size*8/10)
val_size = int(sample_size*1/10)
te_size = int(sample_size*1/10)

train_val_size = val_size+tr_size

#Get the train, validation, and test data
test_x = inputs[-te_size:]
test_y = outputs[-te_size:]
train_x, val_x, train_y, val_y = train_test_split(inputs[0:train_val_size], outputs[0:train_val_size], test_size=val_size/(tr_size+val_size), random_state=1)

param_m = np.mean(train_x,axis=0)   #9layer param mean
param_sd = np.std(train_x,axis=0)   #9layer param std

##########################
#Load FW_MDN that is comparable to the one in Table VI
##########################
fw_FNN = tf.keras.models.load_model(model_path+"/Saved_Models/fw_FNN/", 
                                    custom_objects={'loss_func':FW_MDN_Loss})


######################
#Constructs inputs and outputs to FW-MDN (9layer data)
#In this block, x is C and y is Vs (While in Section III D of the paper, x is Vs and y is C)
#In data9, columns 8~17 are 9 vs's
######################

#output_cols = [str(i) for i in range(8,17)] #list of column names for outputs
#input_cols = [str(i) for i in range(144-50,144)]#only C serves as inputs
input_cols = list(range(144-50,144)) #only C serves as inputs
output_cols = list(range(8,17)) #list of column names for outputs

inputs = data9[input_cols]
outputs = data9[output_cols]
#convert the data to np.array
inputs = np.array(inputs)
outputs = np.array(outputs)

sample_size = len(inputs)
tr_size = int(sample_size*0.8)
val_size = int(sample_size*0.1)
te_size = int(sample_size*0.1)

train_val_size = val_size+tr_size

#Get the train, validation, and test data
test_x = inputs[-te_size:].copy()
test_y = outputs[-te_size:].copy()
train_x, val_x, train_y, val_y = train_test_split(inputs[0:train_val_size], 
                                                  outputs[0:train_val_size], 
                                                  test_size=val_size/(tr_size+val_size), 
                                                  random_state=1)
den_m = np.mean(train_x,axis=0)
den_sd = np.std(train_x,axis=0)

train_dp = np.append(train_x,train_y,axis=1) #dp: [density,param]
val_dp = np.append(val_x,val_y,axis=1) 
test_dp = np.append(test_x,test_y,axis=1) 

dp_m = np.append(den_m,np.zeros(test_y.shape[1]))
dp_sd = np.append(den_sd,np.ones(test_y.shape[1]))


######################### 
#Code for replying Question 4
#########################

#Set the parameter candidates
hid_layer_list = [
                  (100,50),
                 ]
n_mix_list = [4,3,2,1] #list of num_mixes (number of components in the Gaussian mixture)
b_regu_list = [0.001]
w_regu_list = [0.00001]
activ_list = ['tanh']
loss_list = ['L2']
output_dim = 9
min_val_loss_list = []
loop_cnt = 0


#Report the performances for h's
for loss in loss_list:
    for hidl in hid_layer_list:
        for b_reg in b_regu_list:
            for w_reg in w_regu_list:
                for actv in activ_list:
                    for nm in n_mix_list:
                        tf.set_random_seed(1)
                        FW_MDN = build_mdn(param_m = param_m,
                                          param_sd = param_sd,
                                          fw_MODEL = fw_FNN,
                                          loss_func = loss,
                                          input_dim = 50,
                                          hid_layers = hidl,
                                          num_mixes = nm,
                                          out_dims = output_dim,    
                                          w_regu = w_reg,
                                          b_regu = b_reg,   #Regularizing coefficient for the bias terms
                                          activ = actv)
                                         
                        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30)
                        #early_stop = EarlyStoppingByLossVal(monitor='val_loss', value=13000)
                        tf.set_random_seed(1)
                        history = FW_MDN.fit(x=(train_x-den_m)/den_sd, y=train_dp, shuffle=True, 
                                            batch_size=500, epochs=400, callbacks=[early_stop],
                                            validation_data=((val_x-den_m)/den_sd,val_dp), verbose=2)

                        mdn_output = FW_MDN.predict((val_x-den_m)/den_sd)
                        FWMDN_ypred = predict_with_mdn(mdn_output = mdn_output,
                                                       y_true = val_y,
                                                       mix_num = nm
                                                      )  #FW-MDN outputs
                        val_R2 = r2_score(val_y,FWMDN_ypred).round(3)
                        loop_cnt +=1
                        print(loop_cnt,
                              'Val_R2:',val_R2,'hid_layers:',hidl,"b_regu:",b_reg, 
                              'w_regu:',w_reg,'activation:',actv,"num_mixes:",nm,
                              'loss:', loss,
                              'min_val_loss', round(min(history.history['val_loss']),3),
                              'last_val_loss', round(history.history['val_loss'][-1],3))
                        
                        #report_performance(val_y, FWMDN_ypred) 
                        #print(pd.DataFrame(mdn_output[:,-nm:]).describe())
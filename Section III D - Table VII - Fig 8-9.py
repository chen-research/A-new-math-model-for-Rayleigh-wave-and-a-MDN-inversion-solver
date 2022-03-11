#Section III.D - Table VII - Fig. 8 & 9 (data)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from Geophysics_Package import *

#To limit TensorFlow to a specific set of GPUs
pd.set_option('display.max_columns', None)
data_path = "D:/Large_Files_For_Learning/Project_Result_Data/With_Yang_Zhang/Geophysics/"
model_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/python_codes/Clean_Version/"
figure_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/Submissions/IEEE_TGRS/Revision 2022-02/Revised Drafts/Figures/"
data9 = pd.read_csv(data_path+"layers9_kuanyueshu.csv",header=None) #9 layers data
data9 = data9.sample(frac=1,axis=0,replace=False,random_state=1)  #shuffle the data
data9 = data9.reset_index(drop=True)   #reset the index

#pd.set_option('display.max_columns', None)
#print('Data3 shape before dropping duplicates',data3.shape, 'after duplicates are dropped', data3.drop_duplicates().shape)
print('Data9 shape before dropping duplicates',data9.shape, 'after duplicates are dropped', data9.drop_duplicates().shape)

######################
#fw_FNN
#This block builds and trains the 9layer forward model fw_FNN: vs -> C

#Re-train the model if its validation R2 is below 97%
######################

##### --- data for fw_FNN, ie, the 9layer fw_model: vs -> C
input_cols = list(range(8,17)) #list of column names for outputs, Vs
output_cols = list(range(144-50,144))  #C
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


############### --- Build the 9layer fw_model (fw_FNN)
#Note: Retrain Fw_FNN if its validation R2 is below 97%.
fw_FNN = keras.Sequential([
        keras.layers.Dense(50, activation='relu', kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           bias_regularizer=tf.keras.regularizers.l2(0.001), 
                           name='hid_layer1'),  #The 1st hidden layer with 128 nodes and activation function of 'relu'
        keras.layers.Dense(outputs.shape[1],activation='relu'),                                                                              #The last(hence output) layer in the neural net 
],name='FNN')
fw_FNN.compile(optimizer='adam', loss='MSE',  metrics=['MSE'])


##Set early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10) #Stop if the loss does not improve for 3 consecutive epochs

############### --- Fit fw_model and report performance on test data
#Fit the model
tf.set_random_seed(1)
fw_FNN.fit((train_x-param_m)/param_sd, train_y, epochs=800, 
          validation_data=[(val_x-param_m)/param_sd,val_y],
          batch_size=15000, verbose=2, callbacks=[early_stop])

#Performance on train
train_pred = fw_FNN.predict((train_x-param_m)/param_sd)
a = train_y - np.mean(train_y, axis=0)
R2 = 1 - np.sum((train_pred - train_y)**2)/np.sum(a**2)
print("Train R2:", R2)
R2_by_row = r2_score(train_y.transpose(), train_pred.transpose())
R2_by_col = r2_score(train_y, train_pred)
print('r2_score_by_row is',R2_by_row,'r2_score_by_col is',R2_by_col)

#Performance on val
val_pred = fw_FNN.predict((val_x-param_m)/param_sd)
a = val_y - np.mean(val_y,axis=0)
R2 = 1 - np.sum((val_pred - val_y)**2)/np.sum(a**2)
print("Val R2:", R2)
R2_by_row = r2_score(val_y.transpose(), val_pred.transpose())
R2_by_col = r2_score(train_y, train_pred)
print('r2_score_by_row is',R2_by_row,'r2_score_by_col is',R2_by_col)

#Performance on test
test_pred = fw_FNN.predict((test_x-param_m)/param_sd)
a = val_y - np.mean(test_y,axis=0)
R2 = 1 - np.sum((test_pred - test_y)**2)/np.sum(a**2)
print("Test R2:", R2)
R2_by_row = r2_score(test_y.transpose(), test_pred.transpose())
R2_by_col = r2_score(test_y, test_pred)
print('r2_score_by_row is',R2_by_row,'r2_score_by_col is',R2_by_col)

#Performance with sum of squares as a percentage
SSE = np.sum((test_pred-test_y)**2,axis=1) #sum of squares of error
Energy = np.sum((test_y)**2,axis=1) #density energy
ratio = SSE/Energy
pd_ratio = pd.Series(ratio)
MSE = np.mean(ratio) #Mean of SSE/(density energy)

#Performance with sum of abs as a percentage
SAE = np.sum(np.abs(test_pred-test_y),axis=1) #sum of absolute error
AD = np.sum(np.abs(test_y),axis=1) #Sum of absolute density
AE_ratio = pd.Series(SAE/AD)
MAE = np.mean(AE_ratio) #Mean of SSE/(density energy)
print("Test MSE as a percentage of energy:",MSE)
print("Test MAE as a percentage of absolute density",MAE)
AE_ratio[AE_ratio>1]

#Save the trained fw_FNN Model
'''
tf.keras.models.save_model(model=fw_FNN, 
                           filepath=model_path+"/Saved_Models/fw_FNN_validation_set_approach/", 
                           overwrite=True)

#Load the trained fw_FNN Model
fw_FNN = tf.keras.models.load_model(model_path+"/Saved_Models/fw_FNN_validation_set_approach/", 
                                    custom_objects={'loss_func':FW_MDN_Loss})
'''
no_output = True

#This block figures out how much noise should be added to the training and validation data
#The validation data is added the noise of Unif(-1%,1%)
#The training data is added the noise of Unif(-0.2%,0.2%), Unif(-0.5%,0.5%), Unif(-0.8%,0.8%), Unif(-1%,1%), respectively
#We choose the noise added to the training data so that the model performs best on the noised val data (minimum val loss)
#The FW-MDN is built with hyper-params in Table IV.
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


noise = np.random.uniform(low=-0.01, high=0.01, size=val_x.shape)             
noised_val_x = val_x*(1+noise)
for delta in [0.01,0.008,0.005,0.002]:
    l = -delta
    h = delta
    np.random.seed(1) #Set random seed for numpy random generator
    noise = np.random.uniform(low=l, high=h, size=train_x.shape)
    noised_train_x = train_x*(1+noise)
    tf.set_random_seed(1)
    FW_MDN = build_mdn(param_m = param_m,
                       param_sd = param_sd,
                       fw_MODEL = fw_FNN,
                       loss_func = 'L2',
                       input_dim = 50,
                       hid_layers = (100,50),
                       num_mixes = 4,
                       out_dims = 9,    
                       w_regu = 0.001,
                       b_regu = 0.001,   #Regularizing coefficient for the bias terms
                       activ = 'tanh'
                       )
                        
    history = FW_MDN.fit(x=(noised_train_x-den_m)/den_sd, y=train_dp, shuffle=True, 
                            batch_size=1000, epochs=300,
                            validation_data=((noised_val_x-den_m)/den_sd,val_dp), verbose=0)
    print('delta=',delta, 'min val_loss=', min(history.history['val_loss']))



######################
#Prepare data for FW-MDN in Table VII
#In this block, x is C and y is Vs (While in Section III D of the paper, x is Vs and y is C)
######################

input_cols = list(range(144-50,144)) #C, x
output_cols = list(range(8,17)) #Vs, y

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
#Noise the data. 
#Uniform(-0.008,0.008) <-> validation loss:56000
l = -0.008
h = 0.008
np.random.seed(1)
noise = np.random.uniform(low=l, high=h, size=(sample_size,train_x.shape[1]))
noised_train_x = train_x*(1+noise[0:tr_size])
noised_val_x = val_x*(1+noise[tr_size:train_val_size])

den_m = np.mean(train_x,axis=0)
den_sd = np.std(train_x,axis=0)

train_dp = np.append(train_x,train_y,axis=1) #dp: [density,param]
val_dp = np.append(val_x,val_y,axis=1) 
test_dp = np.append(test_x,test_y,axis=1) 

dp_m = np.append(den_m,np.zeros(test_y.shape[1]))
dp_sd = np.append(den_sd,np.ones(test_y.shape[1]))


#########################
#Table VII
#Build and train the FW-MDN: noised C -> several vs
#Note that here x is C and y is Vs, while in the paper Section III D x is Vs and y is C.
#Note: the reported "Overall R2" and "Single R2" are in fact the performance meausre M defined in Section III C
#########################

FW_MDN = build_mdn(param_m = param_m,
                   param_sd = param_sd,
                   fw_MODEL = fw_FNN,
                   loss_func = 'L2',
                   input_dim = 50,
                   hid_layers = (100,50),
                   num_mixes = 4,
                   out_dims = 9,    
                   w_regu = 0.001,
                   b_regu = 0.00001,   #Regularizing coefficient for the bias terms
                   activ = 'tanh'
                  )
                        
early_stop1 = EarlyStoppingByLossVal(monitor='val_loss', value=46000)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
tf.set_random_seed(1)                        
history = FW_MDN.fit(x=(noised_train_x-den_m)/den_sd, y=train_dp, shuffle=True, 
                     batch_size=1000, epochs=500, callbacks=[early_stop1],
                     validation_data=((noised_val_x-den_m)/den_sd,val_dp), verbose=2)

mdn_output = FW_MDN.predict((test_x-den_m)/den_sd)
FWMDN_ypred = predict_with_mdn(mdn_output = mdn_output,
                               y_true = test_y,
                               mix_num = 4
                               )  #FW-MDN outputs
test_R2 = r2_score(test_y,FWMDN_ypred).round(3)
report_performance(test_y, FWMDN_ypred) 
print(pd.DataFrame(mdn_output[:,-2:]).describe())
                        
#Report the performance on test data noised with various noise type.
np.random.seed(1)
noise0 = 0
noise1 = np.random.normal(loc=0.0, scale=0.0025, size=test_x.shape)
noise2 = np.random.normal(loc=0.0, scale=0.005, size=test_x.shape)
noise3 = np.random.uniform(low=-0.005, high=0.005, size=test_x.shape)
noise4 = np.random.uniform(low=-0.01, high=0.01, size=test_x.shape)
noise_list = [noise0, noise1, noise2, noise3, noise4]
    
for k in range(5):
    noise = noise_list[k]
    noised_x = test_x*(1+noise)
    mdn_output = FW_MDN.predict((noised_x-den_m)/den_sd)
    FWMDN_ypred = predict_with_mdn(mdn_output = mdn_output,
                                    y_true = test_y,
                                    mix_num = 4
                                  )  
    print("Test noise type:",str(k))
    report_performance(test_y, FWMDN_ypred)  
        
    x_pred = fw_FNN.predict((FWMDN_ypred-param_m)/param_sd)
    y_R2 = r2_score(test_y, FWMDN_ypred)
    x_R2 = r2_score(test_x, x_pred)
    print('R2 between fw_FNN.predict(ypred) and x_true:', x_R2)
                        

#Save the Model
#tf.keras.models.save_model(model=fw_FNN, 
#                           filepath=model_path+"/Saved_Models/fw_FNN_SectionIIID/", 
#                           overwrite=True)

tf.keras.models.save_model(model=FW_MDN, 
                           filepath=model_path+"/Saved_Models/FW_MDN_TableVI_Noisedata/", 
                           overwrite=True)


##############################
#This block computes R2 between noised_x (noised C) and x_pred(C_pred)
##############################
for k in range(5):
    noise = noise_list[k]
    noised_x = test_x*(1+noise)
    mdn_output = FW_MDN.predict((noised_x-den_m)/den_sd)
    FWMDN_ypred = predict_with_mdn(mdn_output = mdn_output,
                                    y_true = test_y,
                                    mix_num = 4
                                  )  
    print("Test noise type:",str(k))
    x_pred = fw_FNN.predict((FWMDN_ypred-param_m)/param_sd)
    x_R2 = r2_score(noised_x, x_pred)
    print('R2 between fw_FNN.predict(ypred) and x_true (noised):', x_R2)


########################
#Fig 8 and 9 Data 
#100 noised C, true Vs, and Vs_pred (Finally, only 1 sample is randomly selected)
########################
pd.DataFrame(np.append(test_y[0:100],FWMDN_ypred[0:100],axis=1)).to_csv("fig7_100VS.csv",index=False)
pd.DataFrame(noised_x[0:100]).to_csv("fig8_100noisedC.csv",index=False)
pd.DataFrame(x_pred[0:100]).to_csv("fig8_100Cpred.csv",index=False)
r2_score(noised_x[0:100],x_pred[0:100])
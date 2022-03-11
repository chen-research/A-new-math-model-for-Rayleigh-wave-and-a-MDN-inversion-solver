#This file contains codes for Table VI and produce data for Fig. 6-7
#which are plotted with a different software.
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
data_path = "D:/Large_Files_For_Learning/Project_Result_Data/With_Yang_Zhang/Geophysics/"
model_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/python_codes/Clean_Version/"
figure_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/Submissions/IEEE_TGRS/Revision 2022-02/Revised Drafts/Figures/"
data9 = pd.read_csv(data_path+"layers9_kuanyueshu.csv",header=None) #9 layers data
data9 = data9.sample(frac=1,axis=0,replace=False,random_state=1)  #shuffle the data
data9 = data9.reset_index(drop=True)   #reset the index

#pd.set_option('display.max_columns', None)
print('Data9 shape before dropping duplicates',data9.shape, 'after duplicates are dropped', data9.drop_duplicates().shape)


######################
#This block builds and trains fw-FNN: i.e., the 9layer forward model fw_model: vs -> C
######################

##### --- data for the 9layer fw_model: vs -> C
#input_cols = [str(i) for i in range(8,17)] #list of column names for outputs
#output_cols = [str(i) for i in range(144-50,144)]
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


############### --- Build the 9layer fw_model (fw_FNN)
fw_FNN = keras.Sequential([
        keras.layers.Dense(50, activation='relu', kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           bias_regularizer=tf.keras.regularizers.l2(0.001), 
                           name='hid_layer1'),  #The 1st hidden layer with 128 nodes and activation function of 'relu'
        keras.layers.Dense(outputs.shape[1],activation='relu'),                                                                              #The last(hence output) layer in the neural net 
],name='FNN')
fw_FNN.compile(optimizer='adam', loss='MSE',  metrics=['MSE'])
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

######################
##Constructs inputs and outputs to FW-MDN (9layer data)
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



######################### -- Table VI FW-MDN Test Performances.
#NOTE: Stop training the FW-MDN model if the val_loss is less than 17000. 
#Re-train your FW-MDN if it fails to achieve a val_loss below 17000.

#Build and train MDN_model: C -> several vs
#MDN_model Output: [batch_size, (dim(mu)*num_mixes  dim(sigma)*num_mixes  num_mixes)]
#where the target of each mu is y, sigma is the std of mu, the last num_mixes entries are 
#weights of components in the mixtures

#Note: the reported "Overall R2" and "Single R2" are in fact the performance meausre M defined in Section III C
#########################
#Build and train the FW-MDN with the optimal hyper-parameter values
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
early_stop = EarlyStoppingByLossVal(monitor='val_loss', value=13000)
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15)
history = FW_MDN.fit(x=(train_x-den_m)/den_sd, y=train_dp, shuffle=True, 
                                            batch_size=500, epochs=500, callbacks=[early_stop],
                                            validation_data=((val_x-den_m)/den_sd,val_dp), verbose=2)

#Performance on test samples
mdn_output = FW_MDN.predict((test_x-den_m)/den_sd)
FWMDN_ypred = predict_with_mdn(mdn_output = mdn_output,
                               y_true = test_y,
                               mix_num = 4
                               )  #FW-MDN outputs
#test_R2 = r2_score(test_y,FWMDN_ypred).round(3)
print("Note: The reported R2 equals the measure M defined in the paper.")
report_performance(test_y, FWMDN_ypred) 
#print(pd.DataFrame(mdn_output[:,-nm:]).describe())


########################
#Fig 6 and 7 Data 
#100 true C, true Vs, and Vs_pred  (Finally, only one sample is randomly selected)
########################
x_pred = fw_FNN.predict((FWMDN_ypred-param_m)/param_sd)
pd.DataFrame(np.append(test_y[0:100],FWMDN_ypred[0:100],axis=1)).to_csv("fig5_100VS.csv",index=False)
pd.DataFrame(test_x[0:100]).to_csv("fig6_100C.csv",index=False)
pd.DataFrame(x_pred[0:100]).to_csv("fig6_100Cpred.csv",index=False)
r2_score(test_x[0:100],x_pred[0:100])


########### - For the beginning of Section III.D
#For noised data - uniform noise Unif[-0.5%,0.5%] 
#Show how the model performs when there are uniform noise.
#noised_x = test_x*(1+epsilon), where epsilon~uniform(l,h)
###########
l = -0.005
h = 0.005

#Noise the data
np.random.seed(1)
noise = np.random.uniform(low=l, high=h, size=test_x.shape)
noised_x = test_x*(1+noise)

sample_size = test_x.shape[0]
mdn_output = FW_MDN.predict((noised_x-den_m)/den_sd)
FWMDN_ypred = predict_with_mdn(mdn_output = mdn_output[0:sample_size],
                                                       y_true = test_y[0:sample_size],
                                                       mix_num = 2
                                                      )  
x_pred = fw_FNN.predict((FWMDN_ypred-param_m)/param_sd)
#Show how the model performs
y_R2 = r2_score(test_y[0:sample_size], FWMDN_ypred)
x_R2_true = r2_score(test_x[0:sample_size], x_pred)
x_R2_noised = r2_score(noised_x, x_pred)

print('R2 for y:', y_R2)
print('R2 between fw_model.predict(ypred) and x_true:', x_R2_true)
print('R2 between fw_model.predict(ypred) and x_noised:', x_R2_noised)

report_performance(test_y[0:sample_size], FWMDN_ypred)  
print(pd.DataFrame(mdn_output[:,-2:]).describe())


######################
#This block check the performance of the model if FW-MDN output the mode with largest pdf value
######################
sample_size = val_x.shape[0]
mdn_output = FW_MDN.predict((val_x-den_m)/den_sd)
preds = predict_with_modes(mdn_output = mdn_output[0:sample_size],
                  y_true = val_y[0:sample_size],
                  mix_num = 4
                  )
print('mode_pred')
report_performance(val_y[0:sample_size], preds)


#This file contains the codes for Appendix B -- A toy example with MDN

from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.pyplot as plt
from tensorflow.keras import backend as kb
import mdn  #https://pypi.org/project/keras-mdn-layer/

#To limit TensorFlow to a specific set of GPUs
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
pd.set_option('display.max_columns', None)
figure_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/Drafts/Figures/"

#Define the function for training an ordinary neural net model
def train_model(train_input, train_output, val_input, val_output,
                loss_func = 'L1', #Loss function to be used, "L1L2":error is L1 norm, bias and weights are L2 norms
                activ='tanh', #The activation function
                hid_layers= (25,36), #(1st hidden layer nodes, 2nd hidden layer nodes)
                bias_regu_cosnt= 0.01, #The regularization coeff. for bias terms
                w_regu_const= 0.01, #The regularization coeff. for weights
                epoch_num= 300, #The number of maximum epochs that will performed. 
                early_stop_patience= 5, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                show_process= 0, #If equals 1, show the training process, if equals 0, do not show. 
                weights = [1,1/15,1/0.05,1/15] #The weights of each output variable.                                                                 
               ):
    """
    This function returns a trained neural net model.
    """
    #Normalize the data
    train_m = np.mean(train_input,axis=0)
    train_std = np.std(train_input,axis=0)
    Ntrain_input = (train_input - train_m)/train_std
    Nval_input = (val_input - train_m)/train_std
    
    #Build the model structure
    model = keras.Sequential()
    #####################
    if loss_func == 'L1': 
        for node_num in hid_layers:
            model.add(keras.layers.Dense(node_num, activation=activ, 
                                     kernel_initializer='glorot_uniform', 
                                     bias_initializer='glorot_uniform',
                                     kernel_regularizer=tf.keras.regularizers.l1(w_regu_const),
                                     bias_regularizer=tf.keras.regularizers.l1(bias_regu_cosnt))) 
        def loss_fcn(y_true, y_pred):
            weight_vec = np.ones(39)
            weight_vec[0:9] = weights[0]
            weight_vec[9:19] = weights[1]
            weight_vec[19:29] = weights[2]
            weight_vec[29:39] = weights[3]
            c_loss = kb.abs(y_true-y_pred)*weight_vec
            return c_loss
        model.add(keras.layers.Dense(train_output.shape[1]))
        model.compile(optimizer='adam', loss='MSE', metrics=['MSE'])
    
    ######################   
    elif loss_func == 'L2':
        for node_num in hid_layers:
            model.add(keras.layers.Dense(node_num, activation=activ, 
                                     kernel_initializer='glorot_uniform', 
                                     bias_initializer='glorot_uniform',
                                     kernel_regularizer=tf.keras.regularizers.l2(w_regu_const),
                                     bias_regularizer=tf.keras.regularizers.l2(bias_regu_cosnt)))
    
        def loss_fcn(y_true, y_pred):
            weight_vec = np.ones(39)
            weight_vec[0:9] = weights[0]
            weight_vec[9:19] = weights[1]
            weight_vec[19:29] = weights[2]
            weight_vec[29:39] = weights[3]
            c_loss = kb.mean(kb.square(y_true-y_pred)*weight_vec,axis=-1)
            return c_loss
        
        opt = tf.keras.optimizers.Adam()
        model.add(keras.layers.Dense(train_output.shape[1]))    
        model.compile(optimizer=opt, loss='MSE', metrics=['MSE'])
    
    #train the model
    #Set early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stop_patience) #Stop if the loss does not improve for 3 consecutive epochs
    history = model.fit(Ntrain_input, train_output, epochs=epoch_num, 
                        verbose=show_process, callbacks=[early_stop],
                        validation_data=[Nval_input, val_output],
                        batch_size=2000)
    
    if len(history.history['loss'])==epoch_num:
        print("Number of epochs done equals epoch_num!")
    
    #The training R^2
    train_prediction = model.predict(Ntrain_input)
    a = train_output - np.mean(train_output,axis=0)
    train_R2 = 1 - np.sum((train_prediction - train_output)**2)/np.sum(a**2)
    
    #The validation R^2
    val_prediction = model.predict(Nval_input)
    a = val_output - np.mean(val_output,axis=0)
    val_R2 = 1 - np.sum((val_prediction - val_output)**2)/np.sum(a**2)
    #print("Train R2:", train_R2, "validation R2:", val_R2)
    
    for i in range(train_output.shape[1]):
        a = train_output[:,i] - np.mean(train_output[:,i],axis=0)
        R2 = 1 - np.sum((train_prediction[:,i] - train_output[:,i])**2)/np.sum(a**2)
        print(i, 'Training R2', R2)
    
    for i in range(val_output.shape[1]):
        a = val_output[:,i] - np.mean(val_output[:,i],axis=0)
        R2 = 1 - np.sum((val_prediction[:,i] - val_output[:,i])**2)/np.sum(a**2)
        print(i, 'Validation R2', R2)
    return [model, train_R2, val_R2]


#########################
#Data Generation
#y = x+sin(2*pi*x)+epsilon
#########################
x = np.random.uniform(low=0.0,high=1.0, size=10000)
ep = np.random.uniform(low=-0.1,high=0.1, size=10000)
y = x+0.3*np.sin(2*np.pi*x)+ep
data = pd.DataFrame({'x':x,'y':y})
plt.figure(figsize=(8,6))
plt.scatter(x,y)
plt.xlabel('x',color='white')
plt.ylabel('y',color='white')
plt.tick_params(colors='white')
plt.show()


#This block shows how the NN performs when one-to-many problem is present.
#Hard to train!
#The model predicts x from y: y->x
#Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(np.reshape(x,[-1,1]), 
                                                    np.reshape(y,[-1,1]), 
                                                    test_size=0.1, 
                                                    random_state=48)
train_m = np.mean(train_x, axis=0)
train_std = np.std(train_x, axis=0)



##############
#Table 1: Report model performance for different values of the hyper-parameters
##############

#Set the parameter candidates
hid_layer_list = [(5,10,5)]
b_regu_list = [0.01]
w_regu_list = [0.01]
activ_list = ['tanh']
loss_func_list = ['L2']
weight_list =[[1,1,1,1]]

#Report the performances
for hidl in hid_layer_list:
    for b_regu in b_regu_list:
        for w_regu in w_regu_list:
            for actv in activ_list:
                for lossf in loss_func_list:
                    for w in weight_list:
                        NN, train_R2, val_R2 = train_model(train_y, train_x, test_y, test_x, 
                                        loss_func = lossf, #The loss function
                                        activ=actv, #The activation function
                                        hid_layers= hidl, #(1st hidden layer nodes, 2nd hidden layer nodes)
                                        bias_regu_cosnt= b_regu, #The regularization coeff. for bias terms
                                        w_regu_const= w_regu, #The regularization coeff. for weights
                                        epoch_num= 1000, #The number of maximum epochs that will performed. 
                                        early_stop_patience= 30, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                                        show_process= 0, #If equals 1, show the training process, if equals 0, do not show. 
                                        weights = w
                                        )
                        print("Val_R2:",round(val_R2,3), "Train_R2:",round(train_R2,3), 'hid_layers:',hidl,
                              "b_regu:",b_regu, 'w_regu:',w_regu, 'activation:',actv, 'loss:',lossf,'weights:',w)
                        
FNN_test_pred = NN.predict((test_y-train_m)/train_std)  #pred for x



#Construct and train the mdn: y->x
N_HIDDEN = 15
N_MIXES = 4
OUTPUT_DIMS = 1

mdn_model = keras.Sequential()
mdn_model.add(keras.layers.Dense(N_HIDDEN, activation='tanh'))
mdn_model.add(keras.layers.Dense(N_HIDDEN, activation='tanh'))
mdn_model.add(mdn.MDN(1, N_MIXES))
mdn_model.compile(loss=mdn.get_mixture_loss_func(1, N_MIXES), optimizer=keras.optimizers.Adam())
history = mdn_model.fit(x=(train_y-train_m)/train_std, y=train_x, batch_size=1000, epochs=800, validation_split=0.15)
#Plot the mdn outputs, for each Gaussian mixture distribution we draw one sample.
#mdn_output[i] = [mu, sigma, pi] for the ith sample, where mu and sigma are of same sizes
mdn_output = mdn_model.predict((test_y-train_m)/train_std)
mdn_test_pred = np.apply_along_axis(mdn.sample_from_output, 1, mdn_output, OUTPUT_DIMS, N_MIXES, temp=1.0)
mdn_test_pred = np.reshape(mdn_test_pred,[-1,1])
print('R2',r2_score(test_x,mdn_test_pred))



#Fig.9: Plot both predictions
p_start = 0
p_end = 200

f, ax =plt.subplots(1,2,figsize=(12,4)) 
ax[0].scatter(test_x[p_start:p_end], test_y[p_start:p_end],s=10,color='r',label='true')
ax[0].scatter(FNN_test_pred[p_start:p_end], test_y[p_start:p_end],s=10,color='b',label='ordinary FNN pred')
ax[0].set_title("Predictions from ordinary FNN",fontsize=16)
ax[0].set_xlabel('x',fontsize=15)
ax[0].set_ylabel('y',fontsize=15)
ax[0].tick_params(labelsize=14)
ax[0].legend(loc='upper left',fontsize=15)

ax[1].scatter(test_x[p_start:p_end], test_y[p_start:p_end],s=10,color='r',label='true')
ax[1].scatter(mdn_test_pred[p_start:p_end], test_y[p_start:p_end],s=10,color='b',label='MDN pred')
ax[1].set_title("Predictions from MDN",fontsize=16)
ax[1].set_xlabel('x',fontsize=15)
ax[1].set_ylabel('y',fontsize=15)
ax[1].legend(loc='upper left',fontsize=15)
plt.savefig(figure_path+'Fig9.eps',bbox_inches="tight")
plt.show()
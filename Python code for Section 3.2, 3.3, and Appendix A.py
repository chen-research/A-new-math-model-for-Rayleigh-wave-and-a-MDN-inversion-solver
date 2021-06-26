#This file contains codes for Table 3 and Fig. 5
#It also provides the four samples for Fig. 4, which is plotted with the matlab code.
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from tensorflow.keras import backend as kb
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.special import softmax

from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import layers
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd
from sklearn.cluster import OPTICS
from tensorflow_probability import distributions as tfd

#To limit TensorFlow to a specific set of GPUs
pd.set_option('display.max_columns', None)

data_path = "D:/Data/With_Yang_Zhang/processed_data/"
figure_path = "D:/Learning/Academic_Research/Projects/With_Yang_Zhang/Drafts/Figures/"
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


###########################
#This block defines all the functions
###########################
# --- Define function that builds and train an FNN
def train_model(train_input, train_output, val_input, val_output,
                loss_func = 'L1', #Loss function to be used, "L1L2":error is L1 norm, bias and weights are L2 norms
                activ='tanh', #The activation function
                hid_layers= (25,36), #(1st hidden layer nodes, 2nd hidden layer nodes)
                bias_regu_cosnt= 0.01, #The regularization coeff. for bias terms
                w_regu_const= 0.01, #The regularization coeff. for weights
                epoch_num= 300, #The number of maximum epochs that will performed. 
                early_stop_patience= 5, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                show_process= 0, #If equals 1, show the training process, if equals 0, do not show.                                                
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
    
        
        
        opt = tf.keras.optimizers.Adam()
        model.add(keras.layers.Dense(train_output.shape[1]))    
        model.compile(optimizer=opt, loss='MSE', metrics=['MSE'])
    
    #train the model
    #Set early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stop_patience) #Stop if the loss does not improve for 3 consecutive epochs
    history = model.fit(Ntrain_input, train_output, epochs=epoch_num, 
                        verbose=show_process, callbacks=[early_stop],
                        validation_data=[Nval_input, val_output],
                        shuffle=True,
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

#Loss function of our model, MDN_model: C -> vs (many vs, num_mixes)
def FW_MDN_Loss(param_dim, num_mixes, density_dim, fw_model, param_m, param_sd):
    """
    Construct a loss functions for the MDN layer parametrised by number of mixtures.
    
    Notations in the comments
    ------------
    The notation is mainly dim(mu), dim(pi) used in the comments that explain the code. Here we use an example
    to explain what they are.
    
    %Example1%: Assume that we want to predict param from density, i.e. build param=Model(density), where in one sample, 
    dim(param)=3, dim(density)=50. We also assume that in the sample set one value of density may corresponds 
    to 2 values of param. So we need to use a gaussian mixture of 2 components, 
    then define mu = [mu1 mu2] where mu1 contains 3 entries and so does mu2. Therefore, in this example, we have
    dim(param) = 3
    dim(density) = 50  
    dim(mu1) = dim(mu2) = 3      #The target of mu1 is param, the target of mu2 is param
    dim(mu) = dim(sigma) = 6 
    dim(pi) = 2
    
    Inputs.
    -------
    param_dim:int, the dimension of one sample of param, corresponds to dim(param)=3 in %Example1% 
    num_mixes:int, the number of mixtures, corresponds to dim(pi)=2 in %Example1%
    density_dim:int, dimension of density, corresponds to dim(density)=50 in %Example1%
    fw_model:keras.sequential() model, the forward model such that density = fw_model(param)
    param_m:1darray, the mean of [batch_size, param] (axis=0)
    param_sd:1darray, the sd of [batch_size, param] (axis=0)
    
    Outputs.
    --------
    loss_func:function, the loss function.
    
    Comments.
    ---------
    This is a customized loss function, the real loss function is loss_func. This
    is an example on how other arguments are passed to the loss function.
   
    """
    def loss_func(xy_true, y_pred):
        '''
        Inputs.
        -------
        xy_true:[batch_size, dim(density param)], the true values. x:density, y:param
        y_pred:[batch_size, dim(mu sigma pi)], the prediction from the MDN (Gaussian Mixture)
        
        Outputs.
        --------
        param_err + density_err
            
        '''
        
        x_true = xy_true[:,0:density_dim]  #[batch_size,dim(density)]
        y_true = xy_true[:,density_dim:]   #[batch_size,dim(paramters)]
        # Reshape inputs 
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * param_dim) + num_mixes], name='reshape_ypreds') 
        #Now y_pred.shape is [batch_size, dim(mu sigma pi)] where dim(mu)=num_mixes*dim(mu1)
        
        y_true = tf.reshape(y_true, [-1, param_dim], name='reshape_ytrue')  #[batch_size, mu_size]
        #Now y_true.shape = [batch_size, param_dim]
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * param_dim,
                                                                         num_mixes * param_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')  #out_mu.shape=[batch_size, dim(mu)]
                                                                              #out_pi.shape=[batch_size, dim(pi)]
        
        ###### --- Construct the mixture models.
        #The mixture distribution  of the nth sample in the batch is stored in mixture[n]        
        '''
        #out_pi = [batch_size, pi] 
        #cat contain batch_size distributions. For each dist, the sample space is {0,1,...,K-1}, where K = num_mixes
        #Row n of out_pi determines the nth distribution in cat
        #A=cat.prob(2) is a 1d tensor array, A[n] = probability of 2 in the nth distribition in cat
        #Example
        p = [
             [0.1, 0.4, 0.5],
             [0.15,0.25, 0.6]
            ]
        dist = tfd.Categorical(probs=p) #1st distribution: p(x=0)=0.1,p(x=1)=0.4,p(x=2)=0.5
                                        #2nd distribution: p(x=0)=0.15,p(x=1)=0.25,p(x=2)=0.6
        A = dist.prob(2.9)              #probability of 2.9 in the two distributions in dist
        '''       
        cat = tfd.Categorical(logits=out_pi)  #cat[n] is the categorical distribution (with out_pi[n]) for the nth sample in the batch
                
        component_splits = [param_dim] * num_mixes
        #If component_splits = [2,3,1], then mus[0]=[batch_size,2], mus[1]=[batch_size,3], mus[2]=[batch_size,1]
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)        #mus[i].shape =[batch_size, dim(mu_i)], i =0,1,2,...,num_mixes-1
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)    #sigs[i].shape =[batch_size, dim(sigma_i)], i =0,1,2,...,num_mixes-1
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)] #zip(mus, sigs) = [batch_size, mu sigma]
        '''
        for loc, scale in zip(mus, sigs): 
        first loop: loc = mus[0], scale=sigs[0]
        second loop: loc = mus[1], scale=sigs[1]
        ...
        
        #coll[i] contains num_mixes multivariate normal distributions.
        #coll[i][n] is the multivariate distribution with mean = nth predicted mu_i, and scale = nth predicted sigma_i
        '''
        mixture = tfd.Mixture(cat=cat, components=coll) #creates a mixture distribution
                                                        #cat = [batch_size] of categorical distributions
                                                        #cat[n] = the categorical distribution according to pi[n] (the nth realization of pi in the batch)
                                                        #coll = [batch_size] of multivariate normal distributions
                                                        #coll[n] contain num_mixes Gaussian distributions corresponds to cat[n]
                                                        #mixture = [batch_size] of Guassian_Mix distributions
        
        
        loss = mixture.log_prob(y_true)     #Calculates the probability of y_true, [batch_size]
        loss = tf.negative(loss)
        param_loss = tf.reduce_mean(loss)         #loss = mean(-log probability of y_true)
        
        ##### --- computes the fw_model prediction error
        density_loss = 0
        # Get the layers of fw_model
        layers = [l for l in fw_model.layers]
           
        for i in range(num_mixes):
            den_pred = (mus[i]-param_m)/param_sd    
            for k in range(len(layers)):
                den_pred = layers[k](den_pred)
            den_error = tf.math.square(den_pred - x_true)
            den_error = tf.math.reduce_mean(den_error,axis=1)
            prob = tf.math.exp(out_pi[:,i])/tf.reduce_sum(tf.math.exp(out_pi),axis=1)
            den_error = den_error*prob
            den_error = tf.math.reduce_mean(den_error)
            density_loss = density_loss + den_error
            
        #return (param_loss+denity_loss)
        return (param_loss+density_loss)
        

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return loss_func

#Functions for contructing our model, MDN_model
def sigma_activation(x):
    return keras.backend.sigmoid(x)*0.001 + keras.backend.epsilon()

#This class definition is from the mdn pacakge
class MDN(layers.Layer):
    """A Mixture Density Network Layer for Keras.
    My modifications on the original def is marked with #Modified. Only sigma nodes definition are modified.

    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        with tf.name_scope('MDN'):
            #####################
            #Original
            #self.mdn_mus = layers.Dense(self.num_mix * self.output_dim, name='mdn_mus')  # mix*output vals, no activation
            #self.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')  # mix*output vals exp activation
            
            
            #Modified mu's          
            self.mdn_mus = layers.Dense(self.num_mix * self.output_dim, name='mdn_mus', activation='relu')
            #self.mdn_mus = layers.Dense(self.num_mix * self.output_dim, name='mdn_mus')
            
            #Modified sigma's
            self.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, 
                                  activation=sigma_activation, name='mdn_sigmas')
                      
            #pi nodes
            self.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi')  # mix vals, logits
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)
        super(MDN, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights

    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = layers.concatenate([self.mdn_mus(x),
                                          self.mdn_sigmas(x),
                                          self.mdn_pi(x)],
                                         name='mdn_outputs')
        return mdn_out
    
def build_mdn(param_m,
              param_sd,
              input_dim=50,
              hid_layers = (250,150,150,100,100),
              num_mixes = 5,        #number of Gaussian mixtures
              out_dims = 1,    #target dimension: train_y.shape[1]
              w_regu = 0.0001,   #Regularizing coefficient for the weights
              b_regu = 0.0001,   #Regularizing coefficient for the bias terms
              activ = 'tanh'
              
    ):
    mdn_model = keras.Sequential()
    for hdl in hid_layers:
        mdn_model.add(
                       keras.layers.Dense(hdl, activation=activ, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform',
                       kernel_regularizer=tf.keras.regularizers.l2(w_regu),
                       bias_regularizer=tf.keras.regularizers.l2(b_regu))
        )
    mdn_model.add(MDN(output_dimension=out_dims, num_mixtures=num_mixes))
    mdn_model.compile(loss=FW_MDN_Loss(param_dim=out_dims, 
                                       num_mixes=num_mixes,
                                       density_dim=input_dim,
                                       fw_model=fw_model,
                                       param_m=param_m, 
                                       param_sd=param_sd), 
                      optimizer=keras.optimizers.Adam())
    return mdn_model


def predict_with_mdn(mdn_output,y_true,mix_num=9):
    """
    This function outputs the MDN outputs (mu, i.e. the mean) that is 
    closest to y_true (vs_true).
    
    Inputs.
    --------
    mdn_output:ndarray, [batch_size, mix_num_of_mu mix_num_of_sigma mix_num_of_pi], 
               mdn_output[n] determines the nth Gaussian mixture.
    y_true:ndarray, [batch_size, y]
    mix_num:number of components in the Gaussian mixture.
    
    Outputs.
    --------
    predictions:ndarray, [batch_size, len(y_true)], this is y_pred.
                predictions[n] is the mode closest to y_true[n] among the num_mix modes
                of the nth Gaussian mixture.
    """
    batch_size = y_true.shape[0]
    y_size = y_true.shape[1]
    predictions = np.empty([batch_size,y_size])
    
    for n in range(batch_size):
        mu_n = mdn_output[n,0:y_size*mix_num]
        mu_n = np.reshape(mu_n,[mix_num,y_size])        
        idx = np.argmin(np.sum((mu_n - y_true[n])**2,axis=1))
        predictions[n] = mu_n[idx]
    #print("R2:", r2_score(y_true,predictions))
    return predictions

    
    
def report_performance(y_true,y_pred):
    num = y_true.shape[1]
    R2 = r2_score(y_true, y_pred)
    print("Overall R2:", R2)
    for i in range(num):
        print("Single R2:", i, r2_score(y_true[:,i], y_pred[:,i]))
    
    M = 1 - np.mean((y_true - y_pred)**2/(y_true**2))
    print("Overall M:", M)    
    for i in range(num):
        M = 1 - np.mean(np.abs(y_true[:,i] - y_pred[:,i])/np.abs(y_true[:,i]))
        print("Single M:", i, M)


        
def plot_pair(y1, y2, label1, label2, idx_start, idx_end):
    """
    Plot the two 1darrays, y1 and y2.
    idx_start:int, the beginning index 
    idx_end:int, the ending index
    """    
    plt.figure(figsize=(12,6))
    x = np.arange(idx_start,idx_end)
    plt.scatter(x,y1[idx_start:idx_end],s=10,color='r',label=label1)
    plt.scatter(x,y2[idx_start:idx_end],s=10,color='r',label=label2)
    
    plt.plot(x,y1[idx_start:idx_end],linestyle='--',color='r',label=label1)
    plt.plot(x,y2[idx_start:idx_end],linestyle='--',color='b',label=label2)
    plt.legend(loc='upper left')
    plt.tick_params(colors='white')
    print("r2_score({0},{1})=".format(label1,label2),r2_score(y1[idx_start:idx_end],y2[idx_start:idx_end]))
    plt.show()
    

class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True



######################
#This block gives the results in Table 5, 6 in Appendix A
######################

#Build and train the 3layer FNN, bw_model3: C->vs
# --- Data for the 3 layer bw_model: C to vs
output_cols = list(range(2,5)) #list of column names for outputs
input_cols = list(range(114-50,114))
inputs = data3[input_cols]
outputs = data3[output_cols]

#convert the data to np.array
inputs = np.array(inputs)
outputs = np.array(outputs)

sample_size = len(inputs)
tr_size = int(sample_size*9/10)
val_size = int(sample_size*1/10)

train_val_size = val_size+tr_size
train_x, val_x, train_y, val_y = train_test_split(inputs[0:train_val_size], outputs[0:train_val_size], test_size=val_size/(tr_size+val_size), random_state=48)

param3_m = np.mean(train_x,axis=0)
param3_sd = np.std(train_x,axis=0)

##### --- Train bw_model3
bw_model3, train_R2, val_R2 = train_model(train_x, train_y, val_x, val_y, 
                                        loss_func= 'L2', #The loss function
                                        activ= 'tanh', #The activation function
                                        hid_layers= (150,150,150,100,100), #(1st hidden layer nodes, 2nd hidden layer nodes)
                                        bias_regu_cosnt= 0.001, #The regularization coeff. for bias terms
                                        w_regu_const= 0.001, #The regularization coeff. for weights
                                        epoch_num= 500, #The number of maximum epochs that will performed. 
                                        early_stop_patience= 10, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                                        show_process= 0 #If equals 1, show the training process, if equals 0, do not show. 
                                        )
print("Val_R2:",round(val_R2,3), "Train_R2:",round(train_R2,3))   


#Builds and trains 5layer model bw_model5: C->vs

##### --- Data for the 5 layer bw_model: C to vs
output_cols = list(range(4,9)) #list of column names for outputs
input_cols = list(range(124-50,124))
inputs = data5[input_cols]
outputs = data5[output_cols]

#convert the data to np.array
inputs = np.array(inputs)
outputs = np.array(outputs)

sample_size = len(inputs)
tr_size = int(sample_size*9/10)
val_size = int(sample_size*1/10)

train_val_size = val_size+tr_size
train_x, val_x, train_y, val_y = train_test_split(inputs[0:train_val_size], outputs[0:train_val_size], test_size=val_size/(tr_size+val_size), random_state=48)

param5_m = np.mean(train_x,axis=0)
param5_sd = np.std(train_x,axis=0)

##### --- Train bw_model5
bw_model5, train_R2, val_R2 = train_model(train_x, train_y, val_x, val_y, 
                                        loss_func= 'L2', #The loss function
                                        activ= 'tanh', #The activation function
                                        hid_layers= (150,150,150,100,100), #(1st hidden layer nodes, 2nd hidden layer nodes)
                                        bias_regu_cosnt= 0.001, #The regularization coeff. for bias terms
                                        w_regu_const= 0.001, #The regularization coeff. for weights
                                        epoch_num= 500, #The number of maximum epochs that will performed. 
                                        early_stop_patience= 10, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                                        show_process= 0 #If equals 1, show the training process, if equals 0, do not show. 
                                        )
print("Val_R2:",round(val_R2,3), "Train_R2:",round(train_R2,3))   



#Builds and trains 9layer model bw_model9: C->vs

##### --- Data for the 9 layer bw_model: C to vs
output_cols = list(range(8,17)) #list of column names for outputs
input_cols = list(range(144-50,144))
inputs = data9[input_cols]
outputs = data9[output_cols]

#convert the data to np.array
inputs = np.array(inputs)
outputs = np.array(outputs)

sample_size = len(inputs)
tr_size = int(sample_size*9/10)
val_size = int(sample_size*1/10)

train_val_size = val_size+tr_size
train_x, val_x, train_y, val_y = train_test_split(inputs[0:train_val_size], outputs[0:train_val_size], test_size=val_size/(tr_size+val_size), random_state=48)

param9_m = np.mean(train_x,axis=0)
param9_sd = np.std(train_x,axis=0)

##### --- Train bw_model9
bw_model9, train_R2, val_R2 = train_model(train_x, train_y, val_x, val_y, 
                                        loss_func= 'L2', #The loss function
                                        activ= 'tanh', #The activation function
                                        hid_layers= (400,300,300,300,300), #(1st hidden layer nodes, 2nd hidden layer nodes)
                                        bias_regu_cosnt= 0.001, #The regularization coeff. for bias terms
                                        w_regu_const= 0.001, #The regularization coeff. for weights
                                        epoch_num= 100, #The number of maximum epochs that will performed. 
                                        early_stop_patience= 10, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                                        show_process= 0 #If equals 1, show the training process, if equals 0, do not show. 
                                        )
print("Val_R2:",round(val_R2,3), "Train_R2:",round(train_R2,3))   




######################
#This block builds and trains the 9layer forward model fw_model: vs -> C
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
train_x, val_x, train_y, val_y = train_test_split(inputs[0:train_val_size], outputs[0:train_val_size], test_size=val_size/(tr_size+val_size), random_state=48)

param_m = np.mean(train_x,axis=0)   #9layer param mean
param_sd = np.std(train_x,axis=0)   #9layer param std


############### --- Build the 9layer fw_model (FNN)
fw_model = keras.Sequential([
        keras.layers.Dense(40, activation='tanh', kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           bias_regularizer=tf.keras.regularizers.l2(0.001), 
                           name='hid_layer1'),  #The 1st hidden layer with 128 nodes and activation function of 'relu'
        keras.layers.Dense(100, activation='tanh', kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           bias_regularizer=tf.keras.regularizers.l2(0.001), 
                           name='hid_layer2'),  #The 2nd hidden layer.
        keras.layers.Dense(200, activation='tanh', kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           bias_regularizer=tf.keras.regularizers.l2(0.001), 
                           name='hid_layer3'),  #The 2nd hidden layer..
        keras.layers.Dense(200, activation='tanh', kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           bias_regularizer=tf.keras.regularizers.l2(0.001), 
                           name='hid_layer3'),  #The 2nd hidden layer.
        keras.layers.Dense(outputs.shape[1],activation='relu'),                                                                              #The last(hence output) layer in the neural net 
],name='FNN')
fw_model.compile(optimizer='adam', loss='MSE',  metrics=['MSE'])


#Set the learning rate
def scheduler(epoch,lr):
    if epoch < 10:
        return 0.01
    else:
        return lr * tf.math.exp(-0.01)

lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
##Set early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10) #Stop if the loss does not improve for 3 consecutive epochs

#Tensorboard
!rm -rf ./logs/  # Clear any logs from previous runs
#log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs\\training_statistics\\"
tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


############### --- Fit fw_model and report performance on test data
#Fit the model
fw_model.fit((train_x-param_m)/param_sd, train_y, epochs=800, 
          validation_data=[(val_x-param_m)/param_sd,val_y],
          batch_size=15000, verbose=2, callbacks=[early_stop])

#Performance on train
train_pred = fw_model.predict((train_x-param_m)/param_sd)
a = train_y - np.mean(train_y, axis=0)
R2 = 1 - np.sum((train_pred - train_y)**2)/np.sum(a**2)
print("Train R2:", R2)
R2_by_row = r2_score(train_y.transpose(), train_pred.transpose())
R2_by_col = r2_score(train_y, train_pred)
print('r2_score_by_row is',R2_by_row,'r2_score_by_col is',R2_by_col)

#Performance on val
val_pred = fw_model.predict((val_x-param_m)/param_sd)
a = val_y - np.mean(val_y,axis=0)
R2 = 1 - np.sum((val_pred - val_y)**2)/np.sum(a**2)
print("Val R2:", R2)
R2_by_row = r2_score(val_y.transpose(), val_pred.transpose())
R2_by_col = r2_score(train_y, train_pred)
print('r2_score_by_row is',R2_by_row,'r2_score_by_col is',R2_by_col)

#Performance on test
test_pred = fw_model.predict((test_x-param_m)/param_sd)
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
                                                  random_state=48)
den_m = np.mean(train_x,axis=0)
den_sd = np.std(train_x,axis=0)

train_dp = np.append(train_x,train_y,axis=1) #dp: [density,param]
val_dp = np.append(val_x,val_y,axis=1) 
test_dp = np.append(test_x,test_y,axis=1) 

dp_m = np.append(den_m,np.zeros(test_y.shape[1]))
dp_sd = np.append(den_sd,np.ones(test_y.shape[1]))


#Fig.9
#Illustrate the One-to-Many Phenomenon in the dataset
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
plt.savefig(figure_path+'Fig8.eps',bbox_inches="tight")
plt.show()


######################### -- Table 6 FW-MDN Test Performances.
#NOTE: Rebuild the MDN_model if the val_loss is greater than 18000. 
#Try more times if your MDN_model cannot achieve it.

#Build and train MDN_model: C -> several vs
#MDN_model Output: [batch_size, (dim(mu)*num_mixes  dim(sigma)*num_mixes  num_mixes)]
#where the target of each mu is y, sigma is the std of mu, the last num_mixes entries are 
#weights of components in the mixtures
#########################

#Set the parameter candidates
hid_layer_list = [
                  (400,300,300,300,300),
                 ]
n_mix_list = [2] #list of num_mixes (number of components in the Gaussian mixture)
b_regu_list = [0.00001]
w_regu_list = [0.00001]
activ_list = ['tanh']
output_dim = 9
sample_size = test_y.shape[0]#test_y.shape[0]#1000 #For testing performance

#Report the performances for h's
for i in range(0,1):
    for hidl in hid_layer_list:
        for b_reg in b_regu_list:
            for w_reg in w_regu_list:
                for actv in activ_list:
                    for nm in n_mix_list:
                        MDN_model = build_mdn(param_m = param_m,
                                          param_sd = param_sd,
                                          input_dim=50,
                                          hid_layers = hidl,
                                          num_mixes = nm,
                                          out_dims = output_dim,    
                                          w_regu = w_reg,
                                          b_regu = b_reg,   #Regularizing coefficient for the bias terms
                                          activ = actv,
                                         )
                        early_stop1 = EarlyStoppingByLossVal(monitor='val_loss', value=18000)
                        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
                        history = MDN_model.fit(x=(train_x-den_m)/den_sd, y=train_dp, shuffle=True, 
                                            batch_size=10000, epochs=3000, callbacks=[early_stop1],
                                            validation_data=((val_x-den_m)/den_sd,val_dp), verbose=2)

                        mdn_output = MDN_model.predict((test_x-den_m)/den_sd)
                        FWMDN_ypred = predict_with_mdn(mdn_output = mdn_output[0:sample_size],
                                                       y_true = test_y[0:sample_size],
                                                       mix_num = nm
                                                      )  #FW-MDN outputs
                        test_R2 = r2_score(test_y[0:sample_size],FWMDN_ypred).round(3)
                        print('testR2:',test_R2,'hid_layers:',hidl,"b_regu:",b_reg, 
                              'w_regu:',w_reg,'activation:',actv,"num_mixes:",nm)
                        report_performance(test_y[0:sample_size], FWMDN_ypred[0:sample_size]) 
                        print(pd.DataFrame(mdn_output[:,-nm:]).describe())

######################## -- Fig 5
#Plot four randomly chosen x_pred and test_x
########################
x_pred = fw_model.predict((FWMDN_ypred-param_m)/param_sd)
print('R2 between fw_model(FW-MDN outputs) and C_true:', 
      r2_score(test_x,x_pred)
     )
i1, i2, i3, i4 = np.random.choice(a = range(sample_size), size=4, replace=False)
x1 = x_pred[i1]
x2 = x_pred[i2]
x3 = x_pred[i3]
x4 = x_pred[i4]

y1 = test_x[i1]
y2 = test_x[i2]
y3 = test_x[i3]
y4 = test_x[i4]

f, ax =plt.subplots(2,2,figsize=(12,8))    
ax[0,0].plot(np.arange(1,len(x1)+1),x1,color='b',linestyle='-', label=r"$\bf{y}$"+"_pred")
ax[0,0].scatter(np.arange(1,len(x1)+1),y1,color='r',s=8, label=r"$\bf{y}$")
#ax[0,0].set_xlabel('frequency',size='large')
ax[0,0].tick_params(labelsize=14)
ax[0,0].legend(loc='upper left',fontsize=15)


ax[0,1].plot(np.arange(1,len(x1)+1),x2,color='b',linestyle='-',label=r"$\bf{y}$"+"_pred")
ax[0,1].scatter(np.arange(1,len(x1)+1),y2,color='r',s=8,label=r"$\bf{y}$")
#ax[0,1].set_xlabel('frequency',size='large')
ax[0,1].tick_params(labelsize=14)
ax[0,1].legend(loc='upper left',fontsize=15)


ax[1,0].plot(np.arange(1,len(x1)+1),x3,color='b',linestyle='-',label=r"$\bf{y}$"+"_pred")
ax[1,0].scatter(np.arange(1,len(x1)+1),y3,color='r',s=8,label=r"$\bf{y}$")
ax[1,0].set_xlabel('frequency',size=15)
ax[1,0].tick_params(labelsize=14)
ax[1,0].legend(loc='upper left',fontsize=15)


ax[1,1].plot(np.arange(1,len(x1)+1),x4,color='b',linestyle='-',label=r"$\bf{y}$"+"_pred")
ax[1,1].scatter(np.arange(1,len(x1)+1),y4,color='r',s=8,label=r"$\bf{y}$")
ax[1,1].set_xlabel('frequency',size=15)
ax[1,1].tick_params(labelsize=14)
ax[1,1].legend(loc='upper left',fontsize=15)

plt.savefig(figure_path+"Fig6.eps",bbox='tight')
plt.show()
idx_list = [i1,i2,i3,i4]
pd.DataFrame(np.append(test_y[idx_list],FWMDN_ypred[idx_list],axis=1)).to_csv("resforfig5.csv",index=False)


########### - For the beginning of Section 3.4
#For noised data - uniform noise Unif[-0.5%,0.5%] 
#Show how the model performs when there are uniform noise.
#noised_x = test_x*(1+epsilon), where epsilon~uniform(l,h)
###########
l = -0.005
h = 0.005

#Noise the data
noise = np.random.uniform(low=l, high=h, size=test_x.shape)
noised_x = test_x*(1+noise)

sample_size = test_x.shape[0]
mdn_output = MDN_model.predict((noised_x-den_m)/den_sd)
FWMDN_ypred = predict_with_mdn(mdn_output = mdn_output[0:sample_size],
                                                       y_true = test_y[0:sample_size],
                                                       mix_num = nm
                                                      )  
x_pred = fw_model.predict((FWMDN_ypred-param_m)/param_sd)
#Show how the model performs
y_R2 = r2_score(test_y[0:sample_size], FWMDN_ypred)
x_R2_true = r2_score(test_x[0:sample_size], x_pred)
x_R2_noised = r2_score(noised_x, x_pred)

print('R2 for y:', y_R2)
print('R2 between fw_model.predict(ypred) and x_true:', x_R2_true)
print('R2 between fw_model.predict(ypred) and x_noised:', x_R2_noised)

report_performance(test_y[0:sample_size], FWMDN_ypred)  
print(pd.DataFrame(mdn_output[:,-nm:]).describe())

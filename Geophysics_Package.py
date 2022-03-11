###########################
#This block defines all the needed functions
###########################
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import multivariate_normal



def add_percentile_label(df, var_name='R2', label_num=10):
    """
    This function sort df according to var, then adds the column of percentile labels to df.
    nan will remain and be ignored.
    
    Inputs.
    --------
    var_name:str, column name of the variable to be sorted on.
    df:pandas.DataFrame, the dataframe.
    label_num:int, the number of groups to be sorted into.
    
    Outputs.
    --------
    df_aug:pd.DataFrame, the label column is added to df and we get df_aug.
    """
    var = np.array(df[var_name])
    labels = np.zeros(len(var))
    labels[:] = np.nan
    df_aug = df.copy()
    for i in range(1,int(label_num+1)):
        low_percentile = (100/label_num)*(i-1)
        high_percentile = (100/label_num)*i
        indicator = (var<np.nanpercentile(var,high_percentile)) & (var>=np.nanpercentile(var,low_percentile))
        indices = np.where(indicator)[0]
        labels[indices] = i
    df_aug[var_name+'_grp_label'] = labels
    return df_aug

def sort_portfolios(df, var='alpha', windowL=24, min_obs=12, sort_gap=12, finalSort_month=1, port_num=10):
    """
    This function sort funds based on avg of past var, and form portfolios.
    Note that the label of portfolio number is updated on each sort date and will
    be copied to the next sort_gap months til, but not including, the next sort month.
    Also, each sort is based on data in the past window excluding the sort month. 
    
    Inputs.
    ---------
    df:pd.DataFrame, contains the columns of 'caldt', 'group_d', var
    var:str, the column of the variable based on which the avg in the past window is computed.
    windowL:int, the length of the past window over which the avg of var is computed.
                 For the sort that happen at month t, the window for computing past 
                 avg is [t-windowL, t-1].
    min_obs:int, the minimum number of observations required to compute the avg.
    sort_gap:int, If a sort happen at month t, then the next sort happens at t+sort_gap.
    finalSort_month:int, the month when the final sort happens.
    
    Take the default parameter values as an example (suppose the last date in df is "2020-06-30").
    The last sort happens at "2020-01-31", for which the alpha avg in "2018-01-31" to "2019-12-31"
    is computed, the second-to-last sort happens at "2019-01-31".
    
    Note that caldt in df should be monthend.
    
    Outputs.
    ----------
    label_df:pd.DataFrame, with columns 'group_id' (fund id) 
                                        'caldt', 
                                        var+'grp_label' (the label of the fund at caldt)
    
    """
    first_dt = df.caldt.sort_values().iloc[0]
    end_year = df.caldt.sort_values().dt.year.iloc[-1]
    end_dt = dt.datetime(end_year,finalSort_month,1)+MonthEnd(0)
    
    #create sort_dat_list
    sort_dat_list = []
    dat = end_dt
    while dat>=first_dt:
        sort_dat_list.append(dat)
        dat = dat - MonthEnd(sort_gap)
    
    #print(sort_dat_list)
    label_df = pd.DataFrame()
    for dat in sort_dat_list:
        win_begin = dat-MonthEnd(windowL)
        win_end = dat-MonthEnd(1)
        #print(win_begin, win_end)
        window_df = df[(df.caldt>=win_begin)&(df.caldt<=win_end)].copy()
        wobs_counts = window_df.groupby(['group_id'])[var].count().reset_index(drop=False)
        wgid_list = list(set(wobs_counts[wobs_counts[var]>=min_obs].group_id))
        wfiltered_df = window_df[window_df['group_id'].isin(wgid_list)]
        if len(wfiltered_df)==0:
            continue
        
        wavg_df = wfiltered_df.groupby(['group_id'])[var].mean().reset_index(drop=False)
        wlabel_df = add_percentile_label(wavg_df, var_name=var, label_num=port_num)
        
        #Replicate the label to [dat+MonthEnd(1), dat+MonthEnd(sort_gap-1)]
        for i in range(sort_gap):
            monthly_labels = wlabel_df 
            monthly_labels['caldt'] = dat+MonthEnd(i)
            label_df = label_df.append(monthly_labels)
    
    return label_df[['group_id','caldt',var+'_grp_label']]
    



# --- Define function that builds and train an FNN
def build_fwFNN(train_input, train_output, val_input, val_output,
                loss_func = 'L1', #Loss function to be used, "L1L2":error is L1 norm, bias and weights are L2 norms
                activ='tanh', #The activation function
                hid_layers= (25,36), #(1st hidden layer nodes, 2nd hidden layer nodes)
                bias_regu_cosnt= 0.01, #The regularization coeff. for bias terms
                w_regu_const= 0.01, #The regularization coeff. for weights
                epoch_num= 300, #The number of maximum epochs that will performed. 
                early_stop_patience= 5, #Early stopping will be applied, this denotes the number of epochs with no improvement in loss that leads to early stop of training.
                show_process= 0, #If equals 1, show the training process, if equals 0, do not show.                                                
                batchsize= 1000,
                print_res=False
               ):
    """
    This function returns a trained fw-FNN.
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
        
        model.add(keras.layers.Dense(train_output.shape[1],activation='relu'))
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
        model.add(keras.layers.Dense(train_output.shape[1],activation='relu'))    
        model.compile(optimizer=opt, loss='MSE', metrics=['MSE'])
    
    #train the model
    #Set early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stop_patience) #Stop if the loss does not improve for 3 consecutive epochs
    history = model.fit(Ntrain_input, train_output, epochs=epoch_num, 
                        verbose=show_process, callbacks=[early_stop],
                        validation_data=[Nval_input, val_output],
                        shuffle=True,
                        batch_size=batchsize)
    
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
    
    if print_res == True:
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
              fw_MODEL,
              loss_func='L2',
              input_dim=50,
              hid_layers = (250,150,150,100,100),
              num_mixes = 5,        #number of Gaussian mixtures
              out_dims = 1,    #target dimension: train_y.shape[1]
              w_regu = 0.0001,   #Regularizing coefficient for the weights
              b_regu = 0.0001,   #Regularizing coefficient for the bias terms
              activ = 'tanh'
    ):
    mdn_model = keras.Sequential()



    if loss_func == 'L1': 
        for hdl in hid_layers:
            mdn_model.add(
                          keras.layers.Dense(hdl, activation=activ, 
                          kernel_initializer='glorot_uniform', 
                          bias_initializer='glorot_uniform',
                          kernel_regularizer=tf.keras.regularizers.l1(w_regu),
                          bias_regularizer=tf.keras.regularizers.l1(b_regu))
                       )
    
    ######################   
    elif loss_func == 'L2':
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
                                       fw_model=fw_MODEL,
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
    
    #M = 1 - np.mean((y_true - y_pred)**2/(y_true**2))
    #print("Overall M:", M)    
    #for i in range(num):
    #    M = 1 - np.mean(np.abs(y_true[:,i] - y_pred[:,i])/np.abs(y_true[:,i]))
    #    print("Single M:", i, M)


        
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






def Gauss_mix_pdf(x,mu,sigma,pi):
    """
    This function computes negative of the Guassian mixture density
    at x, i.e., -pdf(x).
    
    inputs.
    -----------
    x:array,
    pi:array, the array of weights of each Gaussian distribution, sum to 1
    mu:2darray, the array of means. mu[i] is the mean of the ith Gaussian distribution (1darray)
    sigma:2darray, the array of sigmas. sigma[i] is the standard deviation of the ith Gaussian distribution (1darray)
    
    outputs.
    -----------
    -y: negative of pdf(x) 
    """
    mix_num = len(pi)        
    mv_gauss = multivariate_normal(mu[0],np.diag(sigma[0]))
    y = pi[0]*mv_gauss.pdf(x)
    for i in range(1, mix_num):        
        mv_gauss = multivariate_normal(mu[i],np.diag(sigma[i]))
        y = y + pi[i]*mv_gauss.pdf(x)
    
    return -y

def modes_finder(Mu,Sigma,Pi):
    """
    This function finds all the modes corresponding to the Gaussian 
    mixture distribution.
    inputs.
    ------------
    Pi:1darray, Pi[i] is the weight of the ith mixture.
    Mu:2darray, Mu[i] is the mean of the ith mixture (1darray)
    Sigma:2darray, sigma[i] is the sd of the ith mixture (1darray)
        
    outputs.
    ---------
    modes:2darray, same dimension as mu. modes[j] is the jth local mode (1darray)
    """
    mix_num = len(Pi)
    gauss_dim = Mu.shape[1]
    possible_modes = np.empty([mix_num,gauss_dim])
    possible_modes[:] = np.nan
    
    for i in range(mix_num):
        possible_modes[i] = minimize(Gauss_mix_pdf, x0=Mu[i], 
                            args=(Mu,Sigma,Pi), 
                            method='CG', options={'return_all': False}).x
    
    pdf_vals = [Gauss_mix_pdf(possible_modes[i], Mu, Sigma, Pi) for i in range(mix_num)]
    MODE_index = np.argmin(pdf_vals)
    MODE = possible_modes[MODE_index]
    
    return MODE

def predict_with_modes(mdn_output,y_true,mix_num):
    """
    This function finds mix_num modes from the Gaussian mixture with mix_num components.
    Then it outputs the mode with the largest pdf value.
    
    Inputs.
    --------
    mdn_output:ndarray, [batch_size, mix_num_of_mu mix_num_of_sigma mix_num_of_pi], 
               mdn_output[n] determines the nth Gaussian mixture.
    y_true:ndarray, [batch_size, y]
    mix_num:number of components in the Gaussian mixture.
    
    Outputs.
    --------
    predictions:ndarray, [batch_size, len(y_true)], this is y_pred.
                predictions[n] is the mode with largest pdf value among the num_mix modes
                of the nth Gaussian mixture.
    """
    batch_size = y_true.shape[0]
    x_size = y_true.shape[1]
    predictions = np.empty([batch_size,x_size])
    
    for n in range(batch_size):
        mu_n = mdn_output[n,0:x_size*mix_num]
        mu_n = np.reshape(mu_n,[mix_num,x_size])
        
        sigma_n = mdn_output[n,x_size*mix_num:2*x_size*mix_num]
        sigma_n = np.reshape(sigma_n,[mix_num,x_size])
        
        pi_n = softmax(mdn_output[n,-mix_num:])
        
        mode_estimate = modes_finder(mu_n,sigma_n,pi_n) 
        predictions[n] = mode_estimate
        
    #print("R2:", r2_score(y_true,predictions))
    return predictions
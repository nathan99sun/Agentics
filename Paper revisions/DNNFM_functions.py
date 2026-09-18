import tensorflow as tf
import numpy as np
import pandas as pd

#---------------- Classes and Functions for the sparse Neural Network ----------------

class sparse_nn():
    """
    This class is used to create sparse neural network object

    Inputs:
        hidden_l: list of integers, defining the number of hidden units in each hidden layer
        activation_f: list of ['relu', 'lrelu', 'prelu', 'none']
            Applied activation function for each layer.
            Has to be of length len(hidden_l) + 1.
        dropout_rates_tr: Dropout rates at each layer at training
            Has to be of length len(hidden_l) + 1.
        input_s: int; number of observations.
        output_s: int; number of units in the output layer.
        reg_size: int; number of regressors.
        pen_par_w: float; Regularization parameter weights.
        pen_par_b: float; Regularization parameter bias.
        el_net_p: float; Mixing ratio of Ridge and Lasso regression.
            Has to be between 0 and 1.
        max_iter_nc: int
            Number of epochs with no improvement on the validation loss
            before stopping the training.
        max_iter: int; Maximum number of epochs for training neural network.
        opt: string; Optimizer
        learn_rate: float; learning rate.
        use_bias: binary; True, to include bias
    """
    def __init__(self, hidden_l, activation_f, dropout_rates_tr, 
        output_s, reg_size, pen_par_w, pen_par_b, el_net_p, max_iter_nc, max_iter, 
        opt, learn_rate, use_bias, inter_layer):
        
        # Define and assign attributes for network definition
        self.hidden_l = hidden_l
        self.activation_f = activation_f
        self.dropout_rates_tr = dropout_rates_tr
        self.output_s = output_s
        self.reg_size = reg_size
        self.pen_par_w = pen_par_w
        self.pen_par_b = pen_par_b
        self.el_net_p = el_net_p
        self.max_iter_nc = max_iter_nc
        self.max_iter = max_iter
        self.opt = opt
        self.learn_rate = learn_rate
        self.use_bias = use_bias
        self.inter_layer = inter_layer

        # self.dropout_rates_test = [0 for i in dropout_rates_train]

    def build_neural_network(self):
        if (self.pen_par_w == 0.0 and self.pen_par_b == 0.0):
          
          model = tf.keras.Sequential([])
          model.add(tf.keras.layers.Dropout(self.dropout_rates_tr[0]))
            
          for ii in range(len(self.hidden_l)):
              model.add(tf.keras.layers.Dense(units=self.hidden_l[ii], 
              activation=self.activation_f[ii],
              bias_initializer="glorot_uniform",
              use_bias=self.use_bias
              ))
              model.add(tf.keras.layers.Dropout(self.dropout_rates_tr[ii+1]))

          if self.inter_layer:
            # Add layer with regressor size for calculating factor specific covariance matrix
            model.add(tf.keras.layers.Dense(units=self.reg_size,
                activation=self.activation_f[len(self.activation_f)-2],
                bias_initializer='glorot_uniform',
                use_bias=self.use_bias
            ))
            model.add(tf.keras.layers.Dropout(self.dropout_rates_tr[len(self.dropout_rates_tr)-1]))
          
          # Add output layer with size of number of underlying variables
          model.add(tf.keras.layers.Dense(units=self.output_s,
              activation=self.activation_f[len(self.activation_f)-1],
              bias_initializer='glorot_uniform',
              use_bias=self.use_bias
          ))
        
        else:
          model = tf.keras.Sequential([
            #tf.keras.Input(shape=(self.reg_size), dtype=np.float64),
            #tf.keras.layers.InputLayer(input_shape=(None, self.reg_size)),
            #tf.keras.layers.BatchNormalization()
            #tf.keras.layers.Flatten()
          ])
          model.add(tf.keras.layers.Dropout(self.dropout_rates_tr[0]))
            
          for ii in range(len(self.hidden_l)):
              model.add(tf.keras.layers.Dense(units=self.hidden_l[ii], 
              activation=self.activation_f[ii],
              bias_initializer="glorot_uniform",
              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.pen_par_w*self.el_net_p, 
              l2=self.pen_par_w*(1-self.el_net_p)), 
              bias_regularizer=tf.keras.regularizers.l1(self.pen_par_b), use_bias=self.use_bias
              ))
              model.add(tf.keras.layers.Dropout(self.dropout_rates_tr[ii+1]))
            #model.add(tf.keras.layers.BatchNormalization())


          if self.inter_layer:
            # Add layer with regressor size for calculating factor specific covariance matrix
            model.add(tf.keras.layers.Dense(units=self.reg_size,
                activation=self.activation_f[len(self.activation_f)-2],
                bias_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.pen_par_w*self.el_net_p, 
                l2=self.pen_par_w*(1-self.el_net_p)),
                bias_regularizer=tf.keras.regularizers.l1(self.pen_par_b), use_bias=self.use_bias
            ))
            model.add(tf.keras.layers.Dropout(self.dropout_rates_tr[len(self.dropout_rates_tr)-1]))
          
          # Add output layer with size of number of underlying variables
          model.add(tf.keras.layers.Dense(units=self.output_s,
              activation=self.activation_f[len(self.activation_f)-1],
              bias_initializer='glorot_uniform',
              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.pen_par_w*self.el_net_p, 
              l2=self.pen_par_w*(1-self.el_net_p)),
              bias_regularizer=tf.keras.regularizers.l1(self.pen_par_b),
              use_bias=self.use_bias
          ))
        
        #model.add(tf.keras.layers.Reshape([1, -1]))

        self.model = model

    def compile_nn(self):

        if self.opt == 'Adam':
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=self.learn_rate),
                metrics=[tf.metrics.MeanSquaredError(name='mse'), tf.metrics.MeanAbsoluteError(name='mae')])
        elif self.opt == 'RMSprop':
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.optimizers.RMSprop(learning_rate=self.learn_rate),
                metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsoluteError(name='mae')])
        elif self.opt == 'SGD':
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.optimizers.SGD(learning_rate=self.learn_rate),
                metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsoluteError(name='mae')])

#-------------------------------------------

class reg_loss(tf.keras.losses.Loss):
    
    def __init__(self, pen_par, nnet_weights, el_net_pro=1):
        super(reg_loss, self).__init__()
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        r_loss = tf.keras.regularizers.l1_l2(l1=pen_par*el_net_pro, l2=pen_par*(1-el_net_pro))
        self.rloss = r_loss(nnet_weights)

    def call(self, y_true, y_pred):
        mse = self.mse(y_true, y_pred)
        return mse + self.r_loss

#-------------------------------------------

def split_sample(df, train_s = 0.7):
    
    # column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*train_s)]
    val_df = df[int(n*train_s):]

    return train_df, val_df

#-------------------------------------------

def normalize_data(df, split_s = 0.1):
    split_s = 1-split_s
    df_wt = df.iloc[:-1, :]
    test_df = df.iloc[len(df)-1:len(df), :]

    # Split sample in train and valdation
    n = len(df_wt)
    train_df = df_wt[0:int(n*split_s)]
    val_df = df_wt[int(n*split_s):]

    train_mean = df_wt.mean()
    train_std = df_wt.std()

    train_df_n = (train_df - train_mean) / train_std
    val_df_n = (val_df - train_mean) / train_std
    test_df_n = (test_df - train_mean) / train_std
    
    df_n = pd.concat([train_df_n, val_df_n], axis=0)

    return df_n, test_df_n

#-------------------------------------------

def normalize_df(df):

    df_mean = df.mean()
    df_std = df.std()

    df_n = (df - df_mean) / df_std

    return df_n

#-------------------------------------------
"""
def get_cov_cc_nn(neural_net_w, dat, n_par, use_bias):
  num_f = dat.shape[1]
  in_nn = np.cov(dat.T).reshape(num_f, num_f)
  if use_bias:
    nseq = 2
  else:
    nseq = 1
  
  for ii in range(0, n_par, nseq):
    #print(ii)
    if ii == n_par-1-use_bias:
      in_nn = neural_net_w[ii].T @ in_nn @ neural_net_w[ii]
    else:
      in_nn = tf.keras.activations.relu(neural_net_w[ii].T @ in_nn @ neural_net_w[ii])
    
  return in_nn
"""

def get_cov_cc_nn(neural_net_w, dat, n_par, use_bias):
  num_f = dat.shape[1]
  in_nn = np.cov(dat.T).reshape(num_f, num_f)
  if use_bias:
    nseq = 2
    for ii in range(0, n_par, nseq):
      #print(ii)
      if ii == n_par-1-use_bias:
        in_nn = neural_net_w[ii].T @ in_nn @ neural_net_w[ii] #np.cov(neural_net_w[ii+1])
      else:
        in_nn = tf.keras.activations.relu(neural_net_w[ii].T @ in_nn @ neural_net_w[ii] ) #np.cov(neural_net_w[ii+1]))

  else:
    nseq = 1
    for ii in range(0, n_par, nseq):
      #print(ii)
      if ii == n_par-1-use_bias:
        in_nn = neural_net_w[ii].T @ in_nn @ neural_net_w[ii]
      else:
        in_nn = tf.keras.activations.relu(neural_net_w[ii].T @ in_nn @ neural_net_w[ii])
  
  
    
  return in_nn

#-------------------------------------------

# Soft-Thresholding function
def soft_t(z, a):
  t1 = np.sign(z)
  b = np.abs(z) - a
  t2 = b * (b >= 0)
  z_t = t1 * t2
  return z_t

#-------------------------------------------

def thres_cov_resd_aux(resd, N):
  sig_e_samp = np.cov(resd.T)
  
  thet_par = np.empty((N, N))
  thet_par[:] = np.nan

  for ii in range(0, N):
    for jj in range(0, N):
      thet_par[ii, jj] = np.mean(np.abs(resd[:, ii] * resd[:, jj] - sig_e_samp[ii, jj]))

  return sig_e_samp, thet_par

   
def thres_cov_resd(sig_e_samp, thet_par, C, N, T):
   
  rate_thres = np.sqrt((np.log(N))/T)
  # lam = rate_thres * C * np.ones(shape=(N,N))

  lam = rate_thres * C * thet_par
  
  """ 
  sig_e_diag=np.diag(np.diag(sig_e_samp)**(0.5))
  R = np.linalg.inv(sig_e_diag) @ sig_e_samp @ np.linalg.inv(sig_e_diag)
  M = soft_t(R, lam)
  M = M - np.diag(np.diag(M)) + np.eye(N)
  sig_e_hat = sig_e_diag @ M @ sig_e_diag
  """

  sig_e_diag = np.diag(sig_e_samp)
  sig_e_hat = soft_t(sig_e_samp, lam)
  np.fill_diagonal(sig_e_hat, sig_e_diag)

  return sig_e_hat
   

def thres_resd_new(resd, C, N, T):

  sig_e_samp, thet_par = thres_cov_resd_aux(resd, N)
  sig_e_hat = thres_cov_resd(sig_e_samp, thet_par, C, N, T)

  return sig_e_hat

#-------------------------------------------

# Function to calculate the precision matrix of the factor neural network

def sig_inv_f_nnet(sig_f, sig_e):

  p = sig_e.shape[0]
  sig_e_inv = np.linalg.inv(sig_e)
  sig_inv = sig_e_inv - sig_e_inv @ sig_f @ np.linalg.inv(np.eye(p, p) + sig_e_inv @ sig_f) @ sig_e_inv

  return sig_inv

#-------------------------------------------

# Function to calculate the l2-norm of a symmetric positive definite matrix

def l2_norm(sig_m):
  ev_max = np.sqrt(np.max(np.linalg.eig(sig_m.T@sig_m)[0]))
  return ev_max

#-------------------------------------------

# Function to normalize data

def normalize_dat(data):
  
  num_n = data.shape[0]
  X_mean = np.mean(data,axis=0)
  X_std = np.std(data,axis=0)
  Xn = (data - np.repeat(X_mean.reshape(1, -1), num_n, axis=0)) / np.repeat(X_std.reshape(1, -1), num_n, axis=0)

  return Xn, X_mean, X_std

#-------------------------------------------

# Calculate GMVP weights and variance

def w_gmvp(sig, inv_sig):

  Id = np.ones((sig.shape[0], 1))
  w_gmv = (inv_sig @ Id) / (Id.T @ inv_sig @ Id)
  v_gmv = w_gmv.T @ sig @ w_gmv

  return w_gmv, v_gmv

def normalize_dat_sim(data):
  
  data_m = data.mean()
  data_s = data.std()
  
  data_n = (data - data_m) / data_s

  return data_n, data_m, data_s

def cov_sfm(data):
  T, p = data.shape

  var_mean = data.mean(axis=0)
  data = data - np.repeat(var_mean.reshape(1,-1), T, axis=0)                               #demean
  k = 1

    #vars
  n = T-k                                    # adjust effective sample size

  Ymkt = data.mean(axis = 1).reshape(-1,1) #equal-weighted market factor
  covmkt = (data.T @ Ymkt)/n #covariance of original variables with common factor
  varmkt = (Ymkt.T @ Ymkt)/n #variance of common factor
  sigma_sf = (covmkt @ covmkt.T)/varmkt
  sigma_sf[np.logical_and(np.eye(p),np.eye(p))] = ((data.T@data)/n)[np.logical_and(np.eye(p),np.eye(p))]

  return sigma_sf



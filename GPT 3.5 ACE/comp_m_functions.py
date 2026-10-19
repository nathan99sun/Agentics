import numpy as np
import scipy.linalg as sl
import pandas as pd
import numpy.matlib as mt

#---------------- Functions for competing methods ----------------

# GMVP weights

def gmvp_cov_w(cov_est):

    Id = np.ones(shape=(cov_est.shape[0], 1))
    w_est = (np.linalg.inv(cov_est) @ Id) / (Id.T @ np.linalg.inv(cov_est) @ Id)

    return w_est

#################################################################

# Soft-Thresholding function
def soft_t(z, a):
  t1 = np.sign(z)
  b = np.abs(z) - a
  t2 = b * (b >= 0)
  z_t = t1 * t2
  return z_t

def cov_e_poet(resd, C, N, T):
    rate_thres = 1/np.sqrt(N) + np.sqrt((np.log(N))/T)
    # lam = rate_thres * C * np.ones(shape=(N,N))
    
    sig_e_samp = np.cov(resd.T)
    
    
    thet_par = np.empty((N, N))
    thet_par[:] = np.nan
    
    for ii in range(0, N):
        for jj in range(0, N):
            thet_par[ii, jj] = np.mean((resd[:, ii] * resd[:, jj] - sig_e_samp[ii, jj])**2)
    
    lam = rate_thres * C * np.sqrt(thet_par)
    
    """
    sig_e_diag=np.diag(np.sqrt(np.diag(sig_e_samp)))
    R = np.linalg.inv(sig_e_diag) @ sig_e_samp @ np.linalg.inv(sig_e_diag); 
    M = soft_t(R, lam)
    np.fill_diagonal(M, 1)
    sig_e_hat = sig_e_diag @ M @ sig_e_diag
    """

    sig_e_diag = np.diag(sig_e_samp)
    sig_e_hat = soft_t(sig_e_samp, lam)
    np.fill_diagonal(sig_e_hat, sig_e_diag)

    return sig_e_hat

# Single factor model estimation
def static_factor_obs(X, Y):
    lam = np.linalg.inv(X.T @ X) @ X.T @ Y
    return lam

def single_factor_est(F, Y):
    F = F.reshape(-1, 1)
    lam_sf = np.linalg.inv(F.T @ F) @ F.T @ Y
    resd_sf = Y - F @ lam_sf
    cov_sf = lam_sf.T @ np.cov(F.T).reshape(1,1) @ lam_sf + np.diag(np.diag(np.cov(resd_sf.T)))
    Id = np.ones(shape=(Y.shape[1], 1))

    w_sf = (np.linalg.inv(cov_sf) @ Id) / (Id.T @ np.linalg.inv(cov_sf) @ Id)

    ret_ = {'lam_sf': lam_sf, 'cov_sf': cov_sf, 'w_sf': w_sf}

    return ret_

# Fama-French 3-Factor model estimation
def FF_3F_est(F, Y):
    lam_ff_3f = np.linalg.inv(F.T @ F) @ F.T @ Y
    resd_ff_3f = Y - F @ lam_ff_3f
    cov_ff_3f = lam_ff_3f.T @ np.cov(F.T) @ lam_ff_3f + np.diag(np.diag(np.cov(resd_ff_3f.T)))
    Id = np.ones(shape=(Y.shape[1], 1))

    w_ff_3f = (np.linalg.inv(cov_ff_3f) @ Id) / (Id.T @ np.linalg.inv(cov_ff_3f) @ Id)

    ret_ = {'lam_ff_3f': lam_ff_3f, 'cov_ff_3f': cov_ff_3f, 'w_ff_3f': w_ff_3f}

    return ret_

# Approximate Factor Model estimation
def afm_est(Y, NF):

    n = Y.shape[0]
    p = Y.shape[1]
    
    # L'L normalization
    ev_ = sl.eigh(np.cov(Y))

    # Sort eigenvalues in descending order
    indx_ev = ev_[0].argsort()[::-1]
    # Get eigenvectors
    evec = ev_[1][:, indx_ev]

    # Determining Factors
    F = np.sqrt(n) * evec[:, 0:NF]

    # Factorloadings
    L = Y.T @ F/n

    """
    # F'F normalization
    ev_ = sl.eigh(np.cov(Y.T))

    # Sort eigenvalues in descending order
    indx_ev = ev_[0].argsort()[::-1]
    # Get eigenvectors
    evec = ev_[1][:, indx_ev]

    # Determining Factors
    L = np.sqrt(p) * evec[:, 0:NF]
    # Factorloadings
    F = Y @ L/p
    """
    resd = Y - F @ L.T

    ret_ = {'F': F, 'L': L, 'resd': resd}

    return ret_

# Number of factors selection based on IC criteria of Bai and Ng (2002)
def nf_bn(Y, nf_max):

    n = Y.shape[0]
    p = Y.shape[1]

    IC1 = np.empty((nf_max, 1))
    IC1[:] = np.nan
    IC2 = np.empty((nf_max, 1))
    IC2[:] = np.nan
    IC3 = np.empty((nf_max, 1))
    IC3[:] = np.nan

    for ii in range(1, nf_max+1):
        ret_afm = afm_est(Y, ii)

        V = np.mean(ret_afm['resd']**2)
        
        #Information criteria
        #IC1
        IC1[ii-1, 0] = np.log(V) + ii * ((p+n)/(p*n) * np.log((p*n)/(p+n)))
        #IC2
        IC2[ii-1, 0] = np.log(V) + ii * ((p+n)/(p*n) * np.log(min(n,p)))
        #IC3
        IC3[ii-1, 0] = np.log(V) + ii * (np.log(min(n,p)) / (min(n,p)))
    
    ICs = np.empty((3, 2))
    ICs[:] = np.nan

    ICs[0, 0] = IC1.argmin()
    ICs[1, 0] = IC2.argmin()
    ICs[2, 0] = IC3.argmin()

    ICs[0, 1] = IC1[int(ICs[0, 0])]
    ICs[1, 1] = IC2[int(ICs[1, 0])]
    ICs[2, 1] = IC3[int(ICs[2, 0])]

    ret_ = {'num_f': ICs[:,0], 'ICs': ICs[:,1]}

    return ret_

def w_afm(Y, NF_max):

    n = Y.shape[0]
    p = Y.shape[1]

    num_f = nf_bn(Y, NF_max)['num_f']
    num_f = int(num_f[0])
    est_afm = afm_est(Y, num_f)

    F = est_afm['F']
    L = est_afm['L']
    resd = est_afm['resd']


    Id = np.ones(shape=(Y.shape[1], 1))
    cov_afm = L @ np.cov(F.T).reshape(num_f, num_f) @ L.T + np.diag(np.diag(np.cov(resd.T)))
    cov_afm_poet = L @ np.cov(F.T).reshape(num_f, num_f) @ L.T + cov_e_poet(resd, 2, p, n)

    w_afm = (np.linalg.inv(cov_afm) @ Id) / (Id.T @ np.linalg.inv(cov_afm) @ Id)
    w_afm_poet = (np.linalg.inv(cov_afm_poet) @ Id) / (Id.T @ np.linalg.inv(cov_afm_poet) @ Id)

    ret_ = {'num_f': num_f, 'cov_afm': cov_afm, 'cov_afm_poet': cov_afm_poet, 'w_afm': w_afm, 'w_afm_poet': w_afm_poet}

    return ret_


# Ledoit-Wolf linear shrinkage towards single factor model
def cov_LW_factor(Y, k = None):
        #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
        #    None, np.nan or int
        #Post-Condition: Sigmahat dataframe is returned

    # de-mean returns if required
    N,p = Y.shape                      # sample size and matrix dimension
   
    #default setting
    if k is None:
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                    # adjust effective sample size
    
    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     

    #compute shrinkage target
    Ymkt = Y.mean(axis = 1) #equal-weighted market factor
    covmkt = pd.DataFrame(np.matmul(Y.T.to_numpy(),Ymkt.to_numpy()))/n #covariance of original variables with common factor
    varmkt = np.matmul(Ymkt.T.to_numpy(),Ymkt.to_numpy())/n #variance of common factor
    target = pd.DataFrame(np.matmul(covmkt.to_numpy(),covmkt.T.to_numpy()))/varmkt
    target[np.logical_and(np.eye(p),np.eye(p))] = sample[np.logical_and(np.eye(p),np.eye(p))]
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    # diagonal part of the parameter that we call rho 
    rho_diag =  np.sum(np.diag(piMat))
    
    # off-diagonal part of the parameter that we call rho 
    temp = Y*pd.DataFrame([Ymkt for i in range(p)]).T
    covmktSQ = pd.DataFrame([covmkt[0] for i in range(p)])
    v1 = pd.DataFrame((1/n) * np.matmul(Y2.T.to_numpy(),temp.to_numpy())-np.multiply(covmktSQ.T.to_numpy(),sample.to_numpy()))
    roff1 = (np.sum(np.sum(np.multiply(v1.to_numpy(),covmktSQ.to_numpy())))-np.sum(np.diag(np.multiply(v1.to_numpy(),covmkt.to_numpy()))))/varmkt
    v3 = pd.DataFrame((1/n) * np.matmul(temp.T.to_numpy(),temp.to_numpy()) - varmkt * sample)
    roff3 = (np.sum(np.sum(np.multiply(v3.to_numpy(),np.matmul(covmkt.to_numpy(),covmkt.T.to_numpy())))) - np.sum(np.multiply(np.diag(v3.to_numpy()),(covmkt[0]**2).to_numpy()))) /varmkt**2
    rho_off=2*roff1-roff3
    
    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    # compute shrinkage estimator
    sigmahat = shrinkage*target + (1-shrinkage) * sample
    
    return sigmahat.to_numpy()

def w_LW_factor(Y):

    cov_lw = cov_LW_factor(Y)
    Id = np.ones(shape=(Y.shape[1], 1))
    w_LW_factor = (np.linalg.inv(cov_lw) @ Id) / (Id.T @ np.linalg.inv(cov_lw) @ Id)

    ret_ = {'cov_lw': cov_lw, 'w_LW_factor': w_LW_factor}

    return ret_
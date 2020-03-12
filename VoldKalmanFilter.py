# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:02:57 2020

@author: e512481
"""
import numpy as np
from scipy import sparse
from scipy.linalg import pascal
from scipy.sparse import linalg
# solver; from scipy.sparse import linalg #, bicg
# faster: from scikits.umfpack import spsolve

def vkf(y,fs,f,p=None,bw=None,multiorder=None,solver='scipy-spsolve'):
    """
    VKF 2nd Generation Vold-Kalman Order Filtering.
        
    Note: Filter orders > 4 usually result in ill conditioning and should 
    be avoided. The filter bandwidth determination was implemented for 
    arbitrary order but was not verified for orders higher than 3.
    
    Parameters
    ----------
    y : Array, signal 
    fs : Scalar, Sampling frequency
    f : frequency array [f_1,...,f_K] of signal y
    p : p-order filter (typically between 1 or 4), optional
         Every order increases the roll-off by -40dB per decade. 
         By specifying additional lower-order coefficients, 
         zero boundary conditions are added. 
         For instance: p = [2, 0, 1] applies 2nd order filtering and
         forces the envelope amplitude and its first derivative to zero at t_1 
         and t_N.
         The default is order 2.
    bw : Scalar/Array: bandwidth in Hertz specified by bw, optional
         If bw is a scalar, a constant bandwidth is used; 
         If bw is a 1D array with the same length as y, a time-varying 
         instantaneous bandwidth is applied.   
         The default is bw=fs/100.
    multiorder : order of algorithm, optional 
         If multiorder = 0, single-order algorithm. K orders
         are still extracted, but the single-order algorithm is computationally
         less demanding. This is suggested for high sample rates and/or long timeseries.
         The default is Multiorder algorithm.
    solver : 'scikits-spsolve' (faster) or 'scipy-spsolve' (slower)
    Returns
    -------
    x : Arry, complex envelope(s)
    c : Array, phasor(s) c
    r : Array, selectivity(s)


    References: 
    -------
    [1] Vold, H. and Leuridan, J. (1993), High resolution order tracking
        at extreme slew rates, using Kalman tracking filters. Technical 
        Report 931288, Society of Automotive Engineers.
  
    [2] Tuma, J. (2005), Setting the passband width in the Vold-Kalman
        order tracking filter. Proceedings of the International Congress 
        on Sound and Vibration (ICSV12), Lisbon, Portugal.
        
    Credits:
    Adapted to Python by Cyprien Hoelzl, 2019
    Original matlab code by: Maarten van der Seijs, 2010.
  
    """
    #%% Input processing
    if type(p)==type(None):
        p = 2#Default filter order
    if type(bw) ==type(None):
        bw = fs/100   #Default bandwidth: 0.01*fs
    if type(multiorder)==type(None):
        multiord = True   #Default single-order algorithm
    else:
        multiord = multiorder
    silent=False

    if len(np.array(y).shape)>1:
        if np.array(y).shape[1] > np.array(y).shape[0]:    
            y = np.transpose(np.array(y))
    else:
        y = np.array(y)
    if np.size(bw)!=1:    
        if np.array(bw).shape[1] > np.array(bw).shape[0]:
            bw = np.transpose(np.array(bw))
        else:
            bw = np.array(bw)
    if silent ==False:
        print('preparing array ok')
    #Signal
    n_t = len(y)
    
    #Frequency vector(s)
    (n_f, n_ord) = f.shape
    if n_f == 1:
        f1 = np.ones(n_t,1)*f
        raise Warning('error:', f1)
    elif n_f != n_t:
        raise Warning('The array in f should have 1 or {} elements.'.format(n_t))
    
    #Turn frequency vectors into phasor vectors
    c = np.exp(2j*np.pi*np.cumsum(f,axis=0)/fs)
    #Bandwidth vector
    n_bw = np.size(bw)
    if n_bw == 1:
        bw = np.ones((n_t,1))*bw
    elif n_bw != n_t:
        raise Warning('The array bw should have 1 or {} elements.'.format(n_t))
    
    #Relative bandwidth in radians
    phi = np.pi/fs*bw
    #Filter order
    p_tmp = p
    if np.size(p)>1:
        p = max(p)
    p_lo = np.setdiff1d(p_tmp,p)
     
    #%% Construct filter matrix and bandwidth vector
    #Coefficients of p-order difference equation
    P_lu = pascal(p+1,kind='lower',exact=True)
    # Array with 1 -1 1 -1...
    signalt = np.empty((p+1,))
    signalt[::2] = 1
    signalt[1::2] = -1
    coeffs = P_lu[-1,:]*signalt

    #Linear system of difference equations
    A = sparse.spdiags((np.ones((n_t,1))*coeffs).transpose(),np.arange(0,p+1),n_t-p,n_t)

    #Introduce lower order equations to set boundary conditions to zero
    pp = np.size(p_lo)
    A_pre = sparse.csr_matrix((pp,n_t), dtype=None) #sparse(pp,n_t)
    A_pre[:,0:p+1] = P_lu[p_lo,0:p+1]
    A = sparse.vstack([A_pre, A, A_pre[-1::-1,-1::-1]])
    
    #Determination of filter r-value
    s = np.arange(0,p+1)
    sgn = np.power(-1,s)
    
    #Allocate p+1 linear system
    Q = np.zeros((p+1,p+1))
    b = np.zeros((p+1,1))
    
    #Sum of coefficients
    Q[0,:] = np.ones((p+1))
    b[0] = 2**(2*p-1)
    
    #System of p diff. equations
    for i in range(0,p):
        Q[i+1,:] = np.power(s,(2*(i))) * sgn
    
    #Solve for q
    q = np.linalg.solve(Q,b).transpose()*sgn

    #Calculate r
    num = np.sqrt(2)-1
    den = np.zeros((np.size(bw),1))
    for qi in range(0,np.size(q)):
        den = den + 2*q[0,qi]*np.cos((qi)*phi)
    r = np.sqrt(num/den)


    if (den <= 0).any() | (r > (np.sqrt(1/(2*q[0,0]*np.finfo(float).eps)))).any():
        raise Warning('Ill-conditioned B-matrix selectivity bandwidth is too small.')

    #Generate single-order B matrix
    R = sparse.spdiags(r.transpose(),0,n_t,n_t)
    B = (A.dot(R)).transpose().dot(A.dot(R)) + sparse.eye(n_t)

    #Free memory
    del A, A_pre, R, bw, phi, num, den, f
    
    #%% Construct multi-order matrices
    
    if multiord:
        #Construct sparse diagonal part
        nn = n_t*n_ord
        diags = list(range(-p,p+1))
        diags_B = np.zeros((B.shape[0],len(diags)))
        for i in range(0,len(diags)):
            if diags[i]==0:
                diags_B[:,i] = B.diagonal(diags[i])
            elif diags[i]>0:
                diags_B[:-diags[i],i] = B.diagonal(diags[i])
            elif diags[i]<0:
                diags_B[:diags[i],i] = B.diagonal(diags[i])
        diags_B_r= np.tile(diags_B.T,(1,n_ord))

        BB_D = sparse.diags(diags_B_r,diags,shape=(nn,nn))

        #Prepare sparse upper-diagonal part
        bl_U = int((n_ord**2 - n_ord)/2)

        ii_U = np.zeros((n_t,bl_U))
        jj_U = np.zeros((n_t,bl_U))
        cc_U = np.zeros((n_t,bl_U),dtype = np.float64)
        m = 0
        
        #Upper-diagonal part
        for ki in range(0,n_ord):
            for kj in range((ki+1),n_ord):
                ii_U[:,m] = (ki)*n_t + np.arange(0,n_t)
                jj_U[:,m] = (kj)*n_t + np.arange(0,n_t)
                cc_U[:,m] = np.real(np.conj(c[:,ki])*c[:,kj])
                m = m + 1

        #Construct sparse upper-diagonal part
        BB_U = sparse.csr_matrix((cc_U.flatten(),(ii_U.flatten(),jj_U.flatten())),shape=(nn,nn))
        
        #Assemble sparse matrix
        BB = BB_D + BB_U + BB_U.transpose()

        #Construct right-hand side
        cy = (np.conj(c.T.reshape(-1,1)).T*np.tile(y,(n_ord))).T

        #Free memory
        del diags_B, B, BB_D, BB_U, ii_U, jj_U, cc_U    
    
    #     BB = BB.astype(np.complex64)
    
    # B = B.astype(np.complex64)
    #%% Solve)
    if silent==False:
        print('Solving sparse Ax=b, as single precision complex')
    if multiord:
        if solver=='scikits-spsolve':
            from scikits import umfpack
            xx = 2 * umfpack.spsolve(BB,cy)
        else:
            xx = linalg.spsolve(BB,cy,use_umfpack=False)
        # xx = bicg(BB,cy)
        x = 2 * xx.reshape(n_ord,-1).T  
    else:
        x = np.zeros((n_t,n_ord),dtype=np.complex64)
        for ki in range(0,n_ord):
            cy_k = np.conj(c[:,ki])*y
            if solver=='scikits-spsolve':
                from scikits import umfpack
                x[:,ki] = 2 * umfpack.spsolve(B,cy_k)
            else:
                print('dtype B: ',B.dtype)
                # x[:,ki] = 2 * linalg.lgmres(B,cy_k)
                x[:,ki] = 2 * linalg.spsolve(B,cy_k,use_umfpack=False)
    
    return x,c,r

def lil_repeat(S, repeat):
    # row repeat for lil sparse matrix
    # test for lil type and/or convert
    shape=list(S.shape)
    if isinstance(repeat, int):
        shape[0]=shape[0]*repeat
    else:
        shape[0]=sum(repeat)
    shape = tuple(shape)
    new = sparse.lil_matrix(shape, dtype=S.dtype)
    new.data = S.data.repeat(repeat) # flat repeat
    new.rows = S.indices.repeat(repeat)
    return new


###############################################################################
if __name__ == '__main__' :
    #%%   Demo:
    #   Calling VKF without arguments shows a small demonstration of multi-
    #   order filtering with two crossing orders in the presence of white 
    #   noise. 
    #
    #   Example:
    import matplotlib.pyplot as plt
    import time
    fs = 12000
    T = 50
    dt = 1/fs
    t = np.linspace(0,(T),np.int64((T-dt)/dt+1)).transpose()
    N = t.size
  
    # Instationary component 0
    A0 = 2*np.hstack([np.linspace(0.5,1,np.floor(N/2).astype(np.int64())), np.linspace(1,0.5,np.ceil(N/2).astype(np.int64()))])
    f0 = (0.2*fs - 0.1*fs*np.cos(4*np.pi*t/T)).reshape(-1,1)
    phi0 = 2*np.pi*np.cumsum(f0)*dt+3*t/max(t)
    y0 = A0*np.cos(phi0)
    # Instationary component 1
    A1 = np.hstack([np.linspace(0.5,1,np.floor(N/2).astype(np.int64())), np.linspace(1,0.5,np.ceil(N/2).astype(np.int64()))])
    f1 = (0.2*fs - 0.1*fs*np.cos(np.pi*t/T)).reshape(-1,1)
    phi1 = 2*np.pi*np.cumsum(f1)*dt
    y1 = A1*np.cos(phi1)
    
    # Stationary component
    As1 = np.ones(N)
    fs1 = 0.2*fs*np.ones((N,1))
    phis1 = 2*np.pi*np.cumsum(fs1)*dt
    ys1 = As1*np.sin(phis1)
  
    # White noise
    e = 1.5*np.random.rand(np.size(y1))
  
    # Mixed signal
    y = y0+ y1 + ys1 + e
  
    # Perform VKF on periodic components
    p = 2
    bw = fs/12000
    t1=time.time()
    (x,c,r) = vkf(y,fs,np.hstack([f0, f1, fs1]),p=p,bw=bw)
    print('elaspsed time: {}'.format(time.time()-t1))
    xreal = np.real(x*c)
  
    # # Reveal white noise
    w = y-np.sum(xreal,1)
    
    # Amplitude and Phase
    Amplitude = np.abs(x)
    Phase = np.log(x).imag
  
    # # Plot
    fig,ax = plt.subplots(nrows=3)
    ax[0].specgram(y,round(fs/16),Fs=fs)
    ax[1].plot(t,np.vstack((A0,A1,As1)).T,'--')
    ax[1].plot(t,Amplitude)
    ax[1].set_xlim([min(t),max(t)])
    ax[2].plot(t,Phase)
    ax[2].set_xlim([min(t),max(t)])
    ax[2].set_xlabel('Time (s)')
    ax[1].set_ylabel('Amplitude')
    ax[2].set_ylabel('Phase')
    ax[1].legend(['S1','S2','Ss1'],loc='upper right')
    ax[2].legend(['S1','S2','Ss1'],loc='upper right')
  
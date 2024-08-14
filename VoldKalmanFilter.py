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
    x : Arry, complex envelope[s]
    c : Array, phasor[s] c
    r : Array, selectivity[s]


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
    
    #Frequency vector[s]
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
        cc_U = np.zeros((n_t,bl_U),dtype = np.complex128)
        m = 0
        
        #Upper-diagonal part
        for ki in range(0,n_ord):
            for kj in range((ki+1),n_ord):
                ii_U[:,m] = (ki)*n_t + np.arange(0,n_t)
                jj_U[:,m] = (kj)*n_t + np.arange(0,n_t)
                cc_U[:,m] = np.conj(c[:,ki])*c[:,kj]
                m = m + 1

        #Construct sparse upper-diagonal part
        BB_U = sparse.csr_matrix((cc_U.flatten(),(ii_U.flatten(),jj_U.flatten())),shape=(nn,nn))
        
        #Assemble sparse matrix
        BB = BB_D + BB_U + BB_U.getH()

        #Construct right-hand side
        cy = (np.conj(c.T.reshape(-1,1)).T*np.tile(y,(n_ord))).T

        #Free memory
        del diags_B, B, BB_D, BB_U, ii_U, jj_U, cc_U    
    
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
    import matplotlib
    fs = 12000
    T = 50
    dt = 1/fs
    t = np.arange(0,(T)+dt,dt).transpose()
    N = t.size
  
    # Instationary component 0
    A0 = 2*np.hstack([np.linspace(0.5,1,np.floor(N).astype(np.int64()))])#0.02*t+1
    #,np.linspace(1,0.5,np.ceil(N/2).astype(np.int64()))])
    f0 = (fs/16 - fs/100*np.cos(2*np.pi*1.5*t/T)).reshape(-1,1)
    phase0=3*t/max(t)-2
    phi0 = 2*np.pi*np.cumsum(f0)*dt+phase0
    y0 = A0*np.cos(phi0)
        
    # Instationary component 1
    A1 = 0.7+(0.4*np.cos(t*0.04*np.pi*2)*np.sin(t*0.02*np.pi*2))
    #np.hstack([np.linspace(0.5,1,np.floor(N/2).astype(np.int64())), np.linspace(1,0.5,np.ceil(N/2).astype(np.int64()))])
    f1 = (fs/50 - fs/100*np.cos(2*np.pi*t/T/2)).reshape(-1,1)
    phase1 = np.zeros(N)
    phi1 = 2*np.pi*np.cumsum(f1)*dt+phase1
    y1 = A1*np.cos(phi1)
    
    # Stationary component
    As1 = 1*np.ones(N)
    fs1 = 500*np.ones((N,1))
    phaseS1=np.ones(N)*-1
    phis1 = 2*np.pi*np.cumsum(fs1)*dt+phaseS1  #or equiv:(2*np.pi*fs1.T*t).flatten()+phaseS1
    ys1 = As1*np.cos(phis1)
  
    # White noise
    e = 1/0.28875*(np.random.rand(np.size(y1))-0.5)*0.75
  
    # Mixed signal
    y = y0+ y1 + ys1 + e
  
    # Perform VKF on periodic components
    p = 2
    bw = fs/12000*1.5
    t1=time.time()
    (x,c,r) = vkf(y,fs,np.hstack([f0, f1, fs1]),p=p,bw=bw)
    print('elapsed time: {}'.format(time.time()-t1))
    xreal = np.real(x*c)
  
    # # Reveal white noise
    w = y-np.sum(xreal,1)
    
    # Amplitude and Phase
    Amplitude = np.abs(x)
    Phase = np.log(x).imag
    #%
    # import tikzplotlib
    import cycler
    matplotlib.rcParams.update({'font.family':'serif','font.serif':'Times New Roman','font.size':11,'legend.fontsize':10,
                                'text.usetex': False,
                                'mathtext.default': 'regular',
                                'text.latex.preamble':r"\usepackage{lmodern}"})#'font.family':'Serif',"font.sans-serif": ["Modern"]
    # colors = matplotlib.cm.get_cmap('Dark2',5)
    def makecolormap(colorcount = 4):
        steps = np.arange(0,2+2/(colorcount/2-1),2/(colorcount/2-1))
        colorlist = ([plt.cm.Reds(0.2+.4*(i/6*2)) for i in steps] + 
                    [plt.cm.Blues(0.2 +.4*(i/6*2)) for i in steps])
        return  colorlist
    #%% Spectrogram
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color',makecolormap(3))
    fig,ax = plt.subplots(nrows=1,sharex=True,figsize=[5.39749, 1.1], dpi=300)
    ax= [ax]
    ax[0].specgram(y,round(fs/16),Fs=fs)
    # ax[0].plot(t,As1,'k', linewidth = 1, label = '$X_{theory}$')
    # ax[0].plot(t,np.hstack([f0, fs1, f1]),linestyle='--', dashes=(3, 3), label = ['$X_1$', '$X_2$','$X_3$'], 
    #            colors =makecolormap(colorcount = 4)[1])
    ax[0].plot(t,f0,linestyle='--', dashes=(3, 3), label = ['$X_1$'],  color =makecolormap(colorcount = 3)[1])
    ax[0].plot(t,fs1,linestyle='--', dashes=(3, 3), label = ['$X_3$'], color =makecolormap(colorcount = 3)[2])
    ax[0].plot(t,f1,linestyle='--', dashes=(3, 3), label = ['$X_2$'],  color =makecolormap(colorcount = 3)[3])
    ax[0].set_xlim([min(t),max(t)])
    ax[0].set_ylim([0,1000])
    ax[0].set_yticks((0, 500, 1000))
    ax[0].set_yticklabels([0,0.5,1])
    ax[0].set_ylabel('$f$ (kHz)')
    ax[0].set_xticklabels([])
    ax[0].yaxis.set_label_coords(-0.07,0.5)
    fig.savefig('TimeSeries_Synthetic_Specgram.pdf')
    #%% Amplitude
    fig,ax = plt.subplots(nrows=1,sharex=True,figsize=[5.39749,1.4], dpi=300)
    ax= [ax]
    
    ax[0].plot(t,Amplitude[:,0], label = '$X_1$',  color =makecolormap(colorcount = 3)[1])
    ax[0].plot(t,Amplitude[:,2], label = '$X_3$', color =makecolormap(colorcount = 3)[2])
    ax[0].plot(t,Amplitude[:,1], label = '$X_2$',  color =makecolormap(colorcount = 3)[3])
    
    ax[0].plot(t,np.vstack((A0,As1,A1)).T,'k',linestyle='--', dashes=(3, 3), linewidth = 1,label='_nolegend_')
    ax[0].plot(t,As1,'k',linestyle='--', dashes=(3, 3), linewidth = 1, label = '$X_{theory}$')
    
    ax[0].legend([ax[0].get_legend_handles_labels()[0][ax[0].get_legend_handles_labels()[1].index(i)]  for i in ['$X_1$', '$X_2$','$X_3$',
                                                                                                           '$X_{theory}$']],
                 ['$X_1[k]$', '$X_2[k]$','$X_3[k]$','$X_{theory}$'],
               loc='lower left',
               bbox_to_anchor=(0., 1.08, 1., .102), ncol=4, mode="expand", borderaxespad=0.)
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width, box.height-(box.height-box.y0)*0.25])
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('$A_n[k]$ (m/s$^2$)')
    ax[0].set_xlim([min(t),max(t)])
    ax[0].set_yticks([0,1,2])
    ax[0].yaxis.set_label_coords(-0.07,0.5)
    fig.savefig('TimeSeries_Synthetic_Amplitude.pdf')
    #%% Phase
    fig,ax = plt.subplots(nrows=1,sharex=True,figsize=[5.39749, 1.4], dpi=300)
    ax= [ax]
    
    ax[0].plot(t,Phase[:,0], label = '$X_1$',  color =makecolormap(colorcount = 3)[1])
    ax[0].plot(t,Phase[:,2], label = '$X_3$', color =makecolormap(colorcount = 3)[2])
    ax[0].plot(t,Phase[:,1], label = '$X_2$',  color =makecolormap(colorcount = 3)[3])
    
    ax[0].plot(t,np.vstack((phase0,phaseS1,phase1)).T,'k',linestyle='--', dashes=(3, 3), linewidth = 1, label = ['$X_1$', '$X_2$','$X_3$'])
        
    ax[0].set_xlim([min(t),max(t)])
    ax[0].set_xlabel('$kT_s$ (s)')
    ax[0].set_ylabel('$\\theta_n[k]$ (rad)')
    ax[0].set_yticks([-2,-1,0,1])
    ax[0].yaxis.set_label_coords(-0.07,0.5)
    ax[0].set_ylim(-2,1)
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0+(box.height-box.y0)*0.35, box.width, box.height-(box.height-box.y0)*0.35])
    
    fig.savefig('TimeSeries_Synthetic_Phase.pdf')
    #%%% Time series plot
    fig2,ax2 = plt.subplots(nrows=1,figsize=[5.39749, 2.5], dpi=300)
    ax = [ax2]
    ax[0].plot(t,y,label='$y=X_1+X_2+X_3+\eta$',color ='k',linestyle = '-')
    # ax[0].plot(t,y,label='$X_{theory}$',color ='k',linestyle = '--', dashes=(3,4))
    ax[0].plot(t,xreal.sum(axis=1),label='$X_1+X_2+X_3$')#,color ='k')
    # ax[0].plot(t,y0+ y1 + ys1,label='x$=$S1+S2+$X_3$')
    ax[0].plot(t,xreal[:,0],label='$X_1$')
    ax[0].plot(t,xreal[:,2],label='$X_3$')
    ax[0].plot(t,xreal[:,1],label='$X_2$')
    ax[0].set_xlim([min(t),max(t)])
    ax[0].set_ylabel('$A$ (m/s$^2$)')
    ax[0].set_xlabel('$kT_s$ (s)')
    
    ax[0].yaxis.set_label_coords(-0.07,0.5)
    ax[0].legend([ax[0].get_legend_handles_labels()[0][ax[0].get_legend_handles_labels()[1].index(i)]  for i in ['$y=X_1+X_2+X_3+\eta$','$X_1+X_2+X_3$',
                                                                                                                 '$X_1$', '$X_2$','$X_3$']],
               ['$y[k]$','$y_{filt}[k]$','$X_1[k]$', '$X_2[k]$','$X_3[k]$'],
               loc='lower left',
               bbox_to_anchor=(0., 1.05, 1., .102), ncol=5, mode="expand", borderaxespad=0.)
    box = ax[0].get_position()
    ax2.set_position([box.x0, box.y0+0.1, box.width, box.height-(box.height-box.y0)*0.28])
    fig2.savefig('Properties_Synthetic.pdf')    

    #%%% Time series plot zoomed
    fig2,ax = plt.subplots(nrows=1,figsize=[5.39749, 1.8], dpi=300)
    ax = [ax]
    
    ax[0].plot(t,y,color = 'k',linewidth = 1, label='_nolegend_')
    ax[0].plot(t,xreal.sum(axis=1),label='$X_1+X_2+X_3$')
    ax[0].plot(t,xreal[:,0],label='$X_1$')
    ax[0].plot(t,xreal[:,2],label='$X_3$')
    ax[0].plot(t,xreal[:,1],label='$X_2$')
    ax[0].plot(t,y0,color = 'k',linewidth = 1,linestyle = '--', dashes=(3,4), label='_nolegend_')
    ax[0].plot(t,y1,color = 'k',linewidth = 1,linestyle = '--', dashes=(3,4), label='_nolegend_')
    ax[0].plot(t,ys1,color = 'k',linewidth = 1,linestyle = '--', dashes=(3,4), label='_nolegend_')
    ax[0].plot(t,y0+ y1 + ys1,'k',linewidth = 1,linestyle = '--',dashes=(3,4), label='x$=$S1+S2+$X_3$')
    ax[0].set_xlim([31.995,32.0])
    ax[0].set_ylim([-4,4])
    ax[0].set_ylabel('$A$ (m/s$^2$)')
    ax[0].set_xlabel('$kT_s$ (s)')
    ax[0].yaxis.set_label_coords(-0.07,0.5)
    ax[0].ticklabel_format(useOffset=False)
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0+0.15, box.width, box.height-0.15])
    
    fig2.savefig('Properties_Synthetic_zoomed.pdf') 
    #%%% Time series plot zoomed: y shown
    fig2,ax = plt.subplots(nrows=1,figsize=[5.39749, 1.8], dpi=300)
    ax = [ax]
    
    ax[0].plot(t,y,color = 'k',linewidth = 1, label='$y[k]$')
    ax[0].plot(t,xreal.sum(axis=1),label='$y_{filt}[k]$')
    ax[0].plot(t,y0,color = 'k',linewidth = 0,linestyle = '--', dashes=(3,4), label='_nolegend_')
    ax[0].plot(t,y1,color = 'k',linewidth = 0,linestyle = '--', dashes=(3,4), label='_nolegend_')
    ax[0].plot(t,ys1,color = 'k',linewidth = 0,linestyle = '--', dashes=(3,4), label='_nolegend_')
    ax[0].plot(t,y0+ y1 + ys1,'k',linewidth = 1,linestyle = '--',dashes=(3,4), label='$y_{theo}[k]$')
    ax[0].set_xlim([31.995,32.0])
    ax[0].set_ylim([-5,5])
    ax[0].set_ylabel('$A$ (m/s$^2$)')
    ax[0].set_xlabel('$kT_s$ (s)')
    ax[0].yaxis.set_label_coords(-0.07,0.5)
    ax[0].ticklabel_format(useOffset=False)
    
    ax[0].legend([ax[0].get_legend_handles_labels()[0][ax[0].get_legend_handles_labels()[1].index(i)]  for i in ['$y[k]$','$y_{filt}[k]$','$y_{theo}[k]$']],
               ['$y[k]$','$y_{filt}[k]$','$y_{theo}[k]$'],
               loc='lower left',
               bbox_to_anchor=(0., 1.05, 1., .102), ncol=5, mode="expand", borderaxespad=0.)
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0+0.15, box.width, box.height-0.25])
    fig2.savefig('Properties_Synthetic_zoomed2.pdf') 
    #%% Time series plot zoomed: X shown
    fig2,ax = plt.subplots(nrows=1,figsize=[5.39749, 1.8], dpi=300)
    ax = [ax]
    
    # ax[0].plot(t,y,color = 'k',linewidth = 1, label='_nolegend_')
    # ax[0].plot(t,xreal.sum(axis=1),label='$X_1+X_2+X_3$')
    ax[0].plot(t,xreal[:,0],label='$X_1$',color = makecolormap(colorcount = 3)[1])
    ax[0].plot(t,xreal[:,2],label='$X_3$',color = makecolormap(colorcount = 3)[2])
    ax[0].plot(t,xreal[:,1],label='$X_2$',color = makecolormap(colorcount = 3)[3])
    ax[0].plot(t,y0,color = 'k',linewidth = 1,linestyle = '--', dashes=(3,4), label='$X_{i,theo}$')
    ax[0].plot(t,y1,color = 'k',linewidth = 1,linestyle = '--', dashes=(3,4), label='_nolegend_')
    ax[0].plot(t,ys1,color = 'k',linewidth = 1,linestyle = '--', dashes=(3,4), label='_nolegend_')
    # ax[0].plot(t,y0+ y1 + ys1,'k',linewidth = 1,linestyle = '--',dashes=(3,4), label='x$=$S1+S2+$X_3$')
    ax[0].set_xlim([31.995,32.0])
    ax[0].set_ylim([-2,2])
    ax[0].set_ylabel('$A$ (m/s$^2$)')
    ax[0].set_xlabel('$kT_s$ (s)')
    ax[0].yaxis.set_label_coords(-0.07,0.5)
    ax[0].ticklabel_format(useOffset=False)
    
    ax[0].legend([ax[0].get_legend_handles_labels()[0][ax[0].get_legend_handles_labels()[1].index(i)]  for i in ['$X_1$', '$X_2$','$X_3$','$X_{i,theo}$']],
               ['$X_1[k]$', '$X_2[k]$','$X_3[k]$','$X_{i,theo}$'],
               loc='lower left',
               bbox_to_anchor=(0., 1.05, 1., .102), ncol=5, mode="expand", borderaxespad=0.)    
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0+0.15, box.width, box.height-0.25])
    
    fig2.savefig('Properties_Synthetic_zoomed3.pdf') 
    #%% Test different bandwidths     
    # Mixed signal
    y = y0+ y1 + ys1 + e
  
    # Perform VKF on periodic components
    plist = [1,2]
    bwlist = [fs/12000*np.linspace(0.1,2.5,11),
              fs/12000*np.linspace(1,2.5,11)]
    mse=[]
    for p in plist:
        mse1=[]
        for bw in bwlist[p-1]:
            t1=time.time()
            (x,c,r) = vkf(y,fs,np.hstack([f0, f1, fs1]),p=p,bw=bw)
            print('elapsed time: {}'.format(time.time()-t1))
            xreal = np.real(x*c)
    
            # Reveal white noise
            w = y-np.sum(xreal,1) 
            # Error MSE
            y_true= y0+ y1 + ys1 
            # MSE:
            mse2 = np.mean((y_true - np.sum(xreal,1))**2)
            mse1.append(mse2)
            print('MSE: ', mse2)
        mse.append(mse1)
    #%% Plot parvar
    fig = plt.figure(figsize=[4.8,1.8], dpi=300)
    for i in plist:
        plt.plot(bwlist[i-1],mse[i-1],'x-',label='VKF Order='+str(i))
    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(11)
    plt.xlabel('VKF Bandwidth [Hz]')
    plt.ylabel('MSE')
    plt.legend(loc='upper center')
    plt.tight_layout()
    fig.get_axes()[0].set_position([box.x0*1.2, box.y0, box.width * 0.65, box.height])
    plt.legend(loc='upper left',
                 bbox_to_anchor=(1.02, 1))
    plt.savefig('MSE_Synthetic.pdf')    
import numpy as np
from numpy.fft import fft


def getDistProfilePre(x,m):

    if np.shape(x)[1] > np.shape(x)[0]:
        x = np.transpose(x)
    
    [n,nD] = np.shape(x)
    x = np.concatenate((x,np.zeros((n,nD))))
    X = fft(x, axis=0)
    cum_sumx2 = np.cumsum(x**2, axis = 0)
    sumx2 = cum_sumx2[m-1:n,:] - np.concatenate((np.zeros((1,nD)), cum_sumx2[:n-m,:]))
    return X,n,sumx2
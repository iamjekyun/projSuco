import numpy as np
from numpy.fft import fft, ifft

def getDistProfile(X,n,sumx2,subseq):
    # pesky cornercase
    assert(np.shape(subseq)[1] > np.shape(subseq)[0])

    if np.shape(subseq)[1] > np.shape(subseq)[0]:
         subseq = np.transpose(subseq)

    [subLen,nD] = np.shape(subseq)
    
    return _fastfindNN(X,subseq,n,subLen,nD,sumx2)


def _fastfindNN(X,y,n,m,nD,sumx2):
    y = y[::-1,:]
    # not indexing! concatenation!
    y = np.concatenate((y,np.zeros((2*n - m, np.shape(y)[1]))))
    # y[m:2*n,:] = 0
    Y = fft(y,axis=0)
    Z = X*Y
    z = ifft(Z,axis=0)
    sumy2 = np.sum(y**2, axis=0)

    z = np.real(z[m-1:n,:])
    dist = np.zeros(len(sumx2))

    for idx in range(nD):
        dist = dist + sumx2[:,idx]-2*z[:,idx]+sumy2[idx]
    
    return dist
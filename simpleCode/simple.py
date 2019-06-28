import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.io
import math
# Input : 
#   A : input time series (maybe matrix, e.g. for 12-chroma, N X 12 matrix)
#   subLen : interested subsequence length (scalar)

def Simple_self(A, subLen, opt=1):
    
    if np.shape(A)[1] > np.shape(A)[0]:
         A = np.transpose(A)

    exclusionZone = math.floor(subLen/2 + 0.5)

    nD = np.shape(A)[1]
    matrixProfileLength = np.shape(A)[0] - subLen + 1
    matrixProfile = np.zeros(matrixProfileLength)
    MPIndex = np.zeros(matrixProfileLength, dtype=int)

    [X, n, sumx2] = fastfindNNPre(A, subLen)
    subsequence = A[:subLen,:]
    [distanceProfile, currz, dropval, sumy2] = fastfindNN(X, subsequence,n,subLen,nD,sumx2)
    
    firstz = currz.copy()
    distanceProfile[:exclusionZone] = np.inf
    matrixProfile[0] = distanceProfile.min()
    MPIndex[0] = np.argmin(distanceProfile)
    nz = np.shape(currz)[0]
    for i in range(1,matrixProfileLength):
        subsequence = A[i:i+subLen,:]
        sumy2 = sumy2 - dropval**2 + subsequence[-1,:]**2

        for iD in range(nD):
            currz[1:nz,iD] = currz[0:nz-1,iD] + subsequence[-1,iD] * A[subLen:subLen+nz-1,iD] - dropval[iD]*A[:nz-1,iD]
        currz[0,:] = firstz[i,:]

        dropval=subsequence[0,:]

        distanceProfile = np.zeros(len(sumx2))
        for iD in range(nD):
            distanceProfile = distanceProfile + sumx2[:,iD] - 2*currz[:,iD] + sumy2[iD]
            
        exclusionStart = np.maximum(0,i-exclusionZone)
        exclusionEnd = np.minimum(matrixProfileLength, i + exclusionZone)
        distanceProfile[exclusionStart:exclusionEnd] = np.inf

        
        matrixProfile[i] = distanceProfile.min()
        MPIndex[i] = np.argmin(distanceProfile)
    if opt == 1:
        return matrixProfile, MPIndex
    elif opt == 2:
        return matrixProfile,MPIndex, X, n,sumx2
    else:
        return None

def fastfindNNPre(x, m):
    n = np.shape(x)[0]
    nD = np.shape(x)[1]
    # not indexing! concatenation!
    x = np.concatenate((x,np.zeros((n, nD))))
    X = fft(x, axis=0)
    cum_sumx2 = np.cumsum(x**2, axis = 0)
    sumx2 = cum_sumx2[m-1:n] - np.concatenate((np.zeros((1,nD)), cum_sumx2[:n-m]))
    return X,n,sumx2


def fastfindNN(X,y,n,m,nD,sumx2):
    dropval = y[0,:]
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
    return dist, z, dropval, sumy2



if __name__ == '__main__':
    data = scipy.io.loadmat('suco_data/mazurkas.deepChroma.smoothed.mat')
    data = np.array(data['feat'][0]['chroma'])
    data = np.vstack(data)

    [t1,t2]=Simple_self(data,5)
    plt.subplot(211)
    plt.plot(t1)
    plt.subplot(212)
    plt.plot(t2)
    plt.show()

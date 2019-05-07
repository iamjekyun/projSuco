import numpy as np
from numpy.fft import fft, ifft
# Input : 
#   A : input time series (maybe matrix, e.g. for 12-chroma, N X 12 matrix)
#   subLen : interested subsequence length (scalar)

def Simple_self(A, subLen):
    
    if np.shape(A)[1] > np.shape(A)[0]:
         A = np.transpose(A)

    exclusionZone = np.round(subLen/2)
    nD = np.shape(A)[1]

    matrixProfileLength = np.shape(A)[0] - subLen + 1
    matrixProfile = np.zeros(matrixProfileLength)
    MPIndex = np.zeros(matrixProfileLength)

    X, n, sumx2 = fastfindNNPre(A, subLen)

    subsequence = A[:subLen,:]
    [distanceProfile, currz, dropval, sumy2] = fastfindNN(X, subsequence,n,subLen,featureCol,sumx2)
    firstz = currz.copy()

    distanceProfile[:exclusionZone] = np.inf
    [matrixProfile[0], MPIndex[0]] = min(distanceProfile)

    nz = np.shape(currz)[0]

    for i in range(1,matrixProfileLength):
        subsequence = A[i:i+subLen,:]
        sumy2 -= dropval**2 + subsequence[-1,:]**2
        for iD in range(nD):
            currz[1:nz,iD] = currz[0:nz-1,iD] + subsequence[-1,iD] * A[subLen:subLen+nz-1,iD] - dropval[iD]*A[:nz-1,iD]
        currz[0,:] = firstz[i,:]

        dropval=subsequence[0,:]

        distanceProfile = np.zeros(len(sumx2));
        for iD in range(nD):
            distanceProfile += sumx2[:,iD] - 2*currz[:,iD] + sumy2[iD]

        exclusionStart = np.maximum(0,i-exclusionZone)
        exclusionEnd = np.minimum(matrixProfileLength, i + exclusionZone)
        distanceProfile[exclusionStart:exclusionEnd] = np.inf

        [matrixProfile[i], MPIndex[i]] = min(distanceProfile)
    return matrixProfile, MPIndex






def fastfindNNPre(x, m):
    n = np.shape(x)[0]
    nD = np.shape(x)[1]
    x[n:2*n] = 0
    X = fft(x)
    cum_sumx2 = np.cumsum(x**2)
    sumx2 = cum_sumx2[m-1:n] - np.block([[np.zeros(nD)],[cum_sumx2[:n-m]]])
    return X,n,sumx2


def fastfindNN(X,y,n,m,nD,sumx2):
    dropval = y[0,:]
    y = y[::-1,:]
    y[m:2*n,:] = 0

    Y =fft(y)
    Z = X*Y
    z = ifft(Z)

    sumy2 = sum(y**2)

    z = np.real(z[m-1:n,:])
    dist = np.zeros(len(sumx2))

    for idx in range(nD):
        dist += sumx2[:,idx]-2*z[:,idx]+sumy2[idx]

    return dist, z, dropval, sumy2
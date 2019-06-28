import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import math
from summarize.summarize3 import extractChroma
from OTI import OTI
import librosa.display

# Input : 
#   A : input time series (maybe matrix, e.g. for 12-chroma, N X 12 matrix)
#   subLen : interested subsequence length (scalar)

def simple_Fast(A,B, subLen):
    
    if np.shape(A)[1] > np.shape(A)[0]:
         A = np.transpose(A)
         B = np.transpose(B)

    exclusionZone = math.floor(subLen/2 + 0.5)

    nD = np.shape(A)[1]
    matrixProfileLength = np.shape(A)[0] - subLen + 1
    matrixProfile = np.zeros(matrixProfileLength)
    MPIndex = np.zeros(matrixProfileLength, dtype=int)

    [X, n, sumx2] = fastfindNNPre(A, subLen)
    subsequence = B[:subLen,:]
    [_, firstz, _, _] = fastfindNN(X, subsequence,n,subLen,nD,sumx2)
    
    [X, n, sumx2] = fastfindNNPre(B, subLen)
    subsequence = A[:subLen,:]
    [distanceProfile, currz, dropval,sumy2] = fastfindNN(X, subsequence,n,subLen,nD,sumx2)

    matrixProfile[0] = distanceProfile.min()
    MPIndex[0] = np.argmin(distanceProfile)
    nz = np.shape(currz)[0]

    for i in range(1,matrixProfileLength):
        subsequence = A[i:i+subLen,:]
        sumy2 = sumy2 - dropval**2 + subsequence[-1,:]**2

        for iD in range(nD):
            currz[1:nz,iD] = currz[0:nz-1,iD] + subsequence[-1,iD] * B[subLen:subLen+nz-1,iD] - dropval[iD]*B[:nz-1,iD]
        currz[0,:] = firstz[i,:]

        dropval=subsequence[0,:]

        distanceProfile = np.zeros(len(sumx2))
        for iD in range(nD):
            distanceProfile = distanceProfile + sumx2[:,iD] - 2*currz[:,iD] + sumy2[iD]
        
        matrixProfile[i] = distanceProfile.min()
        MPIndex[i] = np.argmin(distanceProfile)
        
    return matrixProfile, MPIndex

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
    data = []
    for i in range(5):
        path = 'testWAV/test'+str(i)+'.wav'
        chroma = extractChroma(path)
        data.append(chroma)
    print('chromagram done!')
    distM = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if i!=j:
                [_,m] = OTI(data[i], data[j])
                distM[i,j] = np.median(simple_Fast(data[i],m,20)[0])
            else:
                distM[i,j] = 0
            print(i,j)

    fig, ax = plt.subplots()
    im = ax.imshow(distM)
    songs = ['let it be','wet dreamz','gu-ae (original)', 'gu-ae (cover)','brown eyed view']
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(songs)))
    ax.set_yticks(np.arange(len(songs)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(songs)
    ax.set_yticklabels(songs)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(songs)):
        for j in range(len(songs)):
            text = ax.text(j, i, round(distM[i, j],2),
                        ha="center", va="center", color="w")

    ax.set_title('distance color map')
    plt.show()
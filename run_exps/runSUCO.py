import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.io
import math
from OTI import OTI
from simpleCode.simple_Fast import simple_Fast
from simpleCode.getDistProfilePre import getDistProfilePre
from summarize.summarize3 import summarize3
from comparing import comparing


def _geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

data = scipy.io.loadmat('suco_data/mazurkas.deepChroma.smoothed.mat')
data = data['feat']


# modifying available hyperparameters
expNum = 10
nSubseq = 5
chromFeat = 12
ssLen = 40

summaries = np.zeros((expNum,nSubseq,chromFeat,ssLen))
profiles = []

for i in range(expNum):
    [thisSS, thisProf] = summarize3(np.vstack((data[i]['chroma'])),ssLen,nSubseq)
    summaries[i,:,:,:]=thisSS
    thisProf.shape = (-1,1)
    profiles.append(thisProf)

comparisons = np.zeros((expNum,expNum,nSubseq))

for i in range(expNum):
    [X,n,sumx2] = getDistProfilePre(np.vstack((data[i]['chroma'])),ssLen)
    for j in range(expNum):
        if i!=j:
            [s,_] = OTI(np.vstack((data[i]['chroma'])), profiles[j])
            thisDist = comparing(X,n,sumx2,summaries[j,:],s[0])
            comparisons[i,j,:] = thisDist
        else:
            comparisons[i,j,:] = np.zeros(nSubseq)
distM = np.zeros((expNum,expNum))


for i in range(expNum):
    for j in range(expNum):
        distM[i,j] = _geo_mean(abs(comparisons[i,j,:]))

plt.imshow(distM)
plt.colorbar()
plt.show()
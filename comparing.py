import numpy as np
from simpleCode.getDistProfile import getDistProfile

def comparing(X,n,sumx2, subsequences, shiftIdx):
    nSubseq = len(subsequences)

    dist = []
    for k in range(nSubseq):
        thisSS = np.roll(subsequences[k],shiftIdx+1,axis=0)
        # print(thisSS)
        dist.append(min(getDistProfile(X,n,sumx2,thisSS)))

    return dist
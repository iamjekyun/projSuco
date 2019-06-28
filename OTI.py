import numpy as np

def OTI(r,s):
    profile1 = np.sum(r,axis = 1)
    profile2 = np.sum(s,axis = 1)

    oti = np.zeros(12)
    for i in range(np.shape(profile2)[0]):
        oti[i] = np.dot(profile1, np.roll(profile2,i+1))
    
    sortedIndexes = np.argsort(-oti)
    newMusic2 = np.roll(s,sortedIndexes[0]+1,axis=0)
    return sortedIndexes, newMusic2
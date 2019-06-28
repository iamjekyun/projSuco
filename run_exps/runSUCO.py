import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.io
import math
from OTI import OTI
from simpleCode.simple_Fast import simple_Fast
from simpleCode.getDistProfilePre import getDistProfilePre
from summarize.summarize3 import summarize3, extractChroma
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
ownDemo = 2

if ownDemo == False:
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




elif ownDemo == True:
    expNum = 9
    summaries = np.zeros((expNum,nSubseq,chromFeat,ssLen))
    profiles = []
    data = []
    for i in range(expNum):
        path = 'testWAV/test'+str(i)+'.wav'
        chroma = extractChroma(path)
        data.append(chroma)
    
    print('chromagram done!')

    for i in range(expNum):
        [thisSS, thisProf] = summarize3(data[i],ssLen,nSubseq)
        summaries[i,:,:,:]=thisSS
        thisProf.shape = (-1,1)
        profiles.append(thisProf)

    comparisons = np.zeros((expNum,expNum,nSubseq))

    for i in range(expNum):
        [X,n,sumx2] = getDistProfilePre(data[i],ssLen)
        for j in range(expNum):
            if i!=j:
                [s,_] = OTI(data[i], profiles[j])
                thisDist = comparing(X,n,sumx2,summaries[j,:],s[0])
                comparisons[i,j,:] = thisDist
            else:
                comparisons[i,j,:] = np.zeros(nSubseq)
            print('Comparison between song '+str(i) +' and ' +str(j) +' done')
    distM = np.zeros((expNum,expNum))


    for i in range(expNum):
        for j in range(expNum):
            distM[i,j] = _geo_mean(abs(comparisons[i,j,:]))

    fig, ax = plt.subplots()
    im = ax.imshow(distM)
    songs = ['let it be','wet dreamz','gu-ae (original)', 'gu-ae (cover)','brown eyed view','like me (original)','like me (parody)','like me (cover1)','like me (cover2)']
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(songs)))
    ax.set_yticks(np.arange(len(songs)))
    # ... and label them with the respective list entries
    ax.tick_params(axis='x',labelsize=6)
    ax.set_xticklabels(songs)
    ax.set_yticklabels(songs)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(songs)):
        for j in range(len(songs)):
            text = ax.text(j, i, round(distM[i, j],2),
                        ha="center", va="center", color="w")

    ax.set_title('distance heat map by SUCO')
    plt.show()


else:
    expNum = 20
    summaries = np.zeros((expNum,nSubseq,chromFeat,ssLen))
    profiles = []
    mat = scipy.io.loadmat('dataYoutubeCovers.mat')
    fullData = mat['trainData'][0]
    songs = fullData['label_str'][:expNum]
    tmp = []
    for i in songs:
        tmp.append(str(i))
    songs = tmp
    data = fullData['data'][:expNum]

    for i in range(expNum):
        [thisSS, thisProf] = summarize3(data[i],ssLen,nSubseq)
        summaries[i,:,:,:]=thisSS
        thisProf.shape = (-1,1)
        profiles.append(thisProf)

    comparisons = np.zeros((expNum,expNum,nSubseq))

    for i in range(expNum):
        [X,n,sumx2] = getDistProfilePre(data[i],ssLen)
        for j in range(expNum):
            if i!=j:
                [s,_] = OTI(data[i], profiles[j])
                thisDist = comparing(X,n,sumx2,summaries[j,:],s[0])
                comparisons[i,j,:] = thisDist
            else:
                comparisons[i,j,:] = np.zeros(nSubseq)
            print('Comparison between song '+str(i) +' and ' +str(j) +' done')
    distM = np.zeros((expNum,expNum))


    for i in range(expNum):
        for j in range(expNum):
            distM[i,j] = _geo_mean(abs(comparisons[i,j,:]))

    fig, ax = plt.subplots()
    im = ax.imshow(distM)
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(songs)))
    ax.set_yticks(np.arange(len(songs)))
    # ... and label them with the respective list entries
    ax.tick_params(axis='x',labelsize=6)
    ax.set_xticklabels(songs)
    ax.set_yticklabels(songs)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(songs)):
        for j in range(len(songs)):
            text = ax.text(j, i, round(distM[i, j],1),
                        ha="center", va="center", color="w", fontsize=5)

    ax.set_title('distance heat map by SUCO')
    plt.show()

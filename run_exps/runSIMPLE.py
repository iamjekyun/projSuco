import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.io
import math
from OTI import OTI
from simpleCode.simple_Fast import simple_Fast
from summarize.summarize3 import extractChroma

ownDemo = True

if ownDemo == False:
    expNum = 15
    data = scipy.io.loadmat('suco_data/mazurkas.deepChroma.smoothed.mat')
    data = data['feat']
    distM = np.zeros((expNum,expNum))
    for i in range(expNum):
        for j in range(expNum):
            if i!=j:
                [_,m] = OTI(np.vstack(np.array(data[i]['chroma'])), np.vstack(np.array(data[j]['chroma'])))
                distM[i,j] = np.median(simple_Fast(np.vstack(data[i]['chroma']), m, 20)[0])
            else:
                distM[i,j] = 0

    plt.imshow(distM)
    plt.colorbar()
    plt.show()

else:
    expNum = 9
    data = []
    for i in range(expNum):
        path = 'testWAV/test'+str(i)+'.wav'
        chroma = extractChroma(path)
        data.append(chroma)
    print('chromagram done!')
    distM = np.zeros((expNum,expNum))
    for i in range(expNum):
        for j in range(expNum):
            if i!=j:
                [_,m] = OTI(data[i], data[j])
                distM[i,j] = np.median(simple_Fast(data[i],m,20)[0])
            else:
                distM[i,j] = 0
            print('Comparison between song '+str(i) +' and ' +str(j) +' done')

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

    ax.set_title('distance heat map by SIMPLE')
    plt.show()
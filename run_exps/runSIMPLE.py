import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.io
import math
from OTI import OTI
from simpleCode.simple_Fast import simple_Fast

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
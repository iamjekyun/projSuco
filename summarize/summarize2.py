import numpy as np
from simple import Simple_self
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.io
import math
# This is just for convenience

def extractChroma(filepath, option = "cens"):
    sr, y = wavfile.read('test.wav')
    y = y[:,0]
    data = (y-np.mean(y))/np.amax(y)
    if option == "cens":
        chroma_cens = librosa.feature.chroma_cens(y=data, sr=sr)
        return chroma_cens
    elif option == "cq":
        chroma_cq = librosa.feature.chroma_cqt(y=data, sr=sr)
        return chroma_cq
    elif option == "stft":
        chroma_stft = librosa.feature.chroma_stft(y=data, sr=sr)
        return chroma_stft
    else:
        raise NameError('None-available option')
    
def summarize2(data, subseqlen, nSubseqs):
    epsln = 1e-5
    subsequences = [0]*nSubseqs

    pitchProfile = np.sum(data,axis=1)
    pitchProfile /= np.max(pitchProfile)
    [MP, MPIndex] = Simple_self(data, subseqlen)
    
    for idx in range(nSubseqs):
        thisSubseqIdx = np.argmin(MP)
        thisNeighbor = MPIndex[thisSubseqIdx]

        begTrivial = int(max(0,round(thisSubseqIdx-subseqlen/4)))
        endTrivial = int(min(len(MP)-1, round(thisSubseqIdx+subseqlen/4)))
        MP[begTrivial:endTrivial] = np.inf

        begTrivial = int(max(0,round(thisNeighbor-subseqlen/4)))
        endTrivial = int(min(len(MP)-1, round(thisNeighbor+subseqlen/4)))
        MP[begTrivial:endTrivial] = np.inf

        thisSeq = data[:,thisSubseqIdx:thisSubseqIdx+subseqlen]
        subsequences[idx] = thisSeq

    return subsequences, pitchProfile


if __name__ == '__main__':
    data = scipy.io.loadmat('../suco_data/mazurkas.deepChroma.smoothed.mat')
    data = np.array(data['feat'][0]['chroma'])
    data = np.vstack(data)
    [subsequences, pitchProfile] = summarize2(data,13,3)
    for i in subsequences:
        librosa.display.specshow(i,y_axis='chroma')
        plt.colorbar()
        plt.show()

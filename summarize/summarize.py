import numpy as np
from simple import Simple_self
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.io
import math
# This is just for convenience

def extractChroma(filepath, option = "cens"):
    sr, y = wavfile.read(filepath)
    y = y[:,0]
    data = (y-np.mean(y))/np.amax(y)
    # trim the data to 1min ~ 2min
    data = data[60*sr:120*sr]
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
    
def summarize(data, subseqlen, nSubseqs):
    epsln = 1e-5
    subsequences = [0]*nSubseqs

    pitchProfile = np.sum(data,axis=1)
    pitchProfile /= np.max(pitchProfile)
    [MP, MPIndex] = Simple_self(data, subseqlen)
    dtype = [('idx', int),('count',int),('dist',float)]
    stats = np.zeros((len(MPIndex),), dtype = dtype)
    stats['idx'] = np.arange(len(MPIndex))

    for i in range(len(MPIndex)):
        stats['count'][MPIndex[i]] += 1
        stats['dist'][MPIndex[i]] += MP[i]

    stats['dist'] = np.divide(stats['dist'], stats['count'] + epsln)

    for idx in range(nSubseqs):
        # Substats contains info who has the maximum count
        subStats = stats[np.where(stats['count'] >= np.nanmax(stats['count']))]
        subStats = np.sort(subStats, order = ['dist','idx'])
        thisIdx = subStats['idx'][0]
        beginTrivial = max(0,round(thisIdx - subseqlen/4))
        endTrivial = min(np.shape(data)[1]-1, round(thisIdx + subseqlen/4))

        thisSeq = data[:,thisIdx:thisIdx+subseqlen]
        subsequences[idx] = thisSeq
        # Setting trivial indices to -1'
        stats['dist'][np.where((beginTrivial <= stats['idx']) & (stats['idx'] <= endTrivial))] = -1
        stats['count'][np.where((beginTrivial <= stats['idx']) & (stats['idx'] <= endTrivial))] = -1
        stats['dist'][np.where(MPIndex == thisIdx)] = -1
        stats['count'][np.where(MPIndex == thisIdx)] = -1
        stats['dist'][MPIndex[thisIdx]] = -1
        stats['count'][MPIndex[thisIdx]] = -1

    return subsequences, pitchProfile


if __name__ == '__main__':
    data = scipy.io.loadmat('suco_data/mazurkas.deepChroma.smoothed.mat')
    data = np.array(data['feat'][0]['chroma'])
    data = np.vstack(data)
    [subsequences, pitchProfile] = summarize(data,13,3)

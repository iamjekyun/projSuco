import numpy as np
from simple import Simple_self
import librosa

# This is just for convenience

def extractChroma(filepath, option = "cens"):
    y, sr = librosa.load(filepath)
    if option == "cens":
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        return chroma_cens
    elif option == "cq":
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        return chroma_cq
    elif option == "stft":
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        return chroma_stft
    else:
        raise NameError('None-available option')
    
def summarize(data, subseqlen, nSubseqs):
    epsln = 1e-5
    subsequences = [0]*nSubseqs

    data = np.reshape(data, (2,-1))
    pitchProfile = np.sum(data,axis=1)
    pitchProfile /= np.max(pitchProfile)
    [MP, MPIndex] = Simple_self(data, subseqlen)
    dtype = [('idx', int),('count',int),('dist',float)]
    stats = np.zeros((len(MPIndex), 3), dtype = dtype)
    stats['idx'] = np.arange(len(MPIndex))

    for i in range(len(MPIndex)):
        stats[MPIndex[i],1] += 1
        stats[MPIndex[i],2] += MP[i]

    
    stats['dist'] = np.divide(stats['dist'], stats['count'] + epsln)

    for idx in range(len(nSubseqs)):
        # Substats contains info who has the maximum count
        subStats = stats[np.where(stats >= np.nanmax(stats['count']))]
        subStats = np.sort(subStats, order = ['dist','idx'])
        thisIdx = subStats[0,0]

        beginTrivial = max(0,thisIdx - subseqlen/4)
        endTrivial = min(np.shape(data)[0], round(thisIdx + subseqlen/4))

        thisSeq = data[:,thisIdx:thisIdx+subseqlen]
        subsequences[idx] = thisSeq

        # Setting trivial indices to -1
        stats[np.where(beginTrivial <= stats['idx'] <= endTrivial), 'dist'] = -1
        stats[np.where(beginTrivial <= stats['idx'] <= endTrivial), 'count'] = -1
        stats[np.where(MPIndex == thisIdx), 1:] = -1
        stats[MPIndex[thisIdx], 1:] = -1

    return subsequences, pitchProfile
import numpy as np
import sys
from scipy.spatial.distance import hamming
def get_similarity(measure, p1, p2):
    if (measure == 'hamming'):
        return hamming_similarity(p1, p2)
    elif (measure == 'cosine'):
        return cosine_similarity(p1, p2)
    elif (measure == 'l1'):
        return l1_similaity(p1, p2)

    return False

def hamming_similarity(p1, p2):
    d = 0
    try:
        a1 = p1.flatten()
        a2 = p2.flatten()
    except ValueError:
        print(a1, a2)
    return hamming(a1, a2)

def cosine_similarity(p1, p2):
    #print(p1.shape, p2.shape)
    return np.dot(p1.T, p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))

def l1_similaity(p1, p2):
    #print(p1)
    return np.sum(np.abs(p1 - p2))

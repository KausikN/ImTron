'''
Distance Functions
'''

# Imports
import numpy as np

# Main Functions
def EuclideanDistance(p1, p2, **params):
    '''
    Euclidean Distance
    '''
    dist_matrix = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
    return dist_matrix

def NormalisedEuclideanDistance(p1, p2, **params):
    '''
    Normalised Euclidean Distance
    '''
    dist_matrix = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
    d_min, d_max = dist_matrix.min(), dist_matrix.max()
    if not (d_max == d_min):
        dist_matrix = (dist_matrix - d_min) / (d_max - d_min)
    return dist_matrix

# Main Vars
DISTANCE_FUNCS = {
    "Euclidean": EuclideanDistance,
    "Normalised Euclidean": NormalisedEuclideanDistance
}
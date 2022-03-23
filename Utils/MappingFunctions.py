'''
Mapping Functions
'''

# Imports
import math
import numpy as np
np.random.seed(0)
from tqdm import tqdm

# Main Functions
# Single Vector Mapping
def Mapping_1V_minDist(Data, **params):
    '''
    Map single-vector list to single-vector list using minimum distance
    '''
    # Data
    L1, L2 = Data["1"], Data["2"]
    # Fix Data
    L1 = np.array(L1)
    L2 = np.array(L2)
    reverse = L1.shape[0] < L2.shape[0]
    if reverse: L1, L2 = L2, L1
    # Params
    tqdm_disable = "tqdm" in params.keys() and (not params["tqdm"])

    # Find Distances
    dist_matrix = params["dist"]["func"](L1, L2, **params["dist"]["params"])
    # Map
    mapping = []
    mapRatio = math.ceil(L1.shape[0] / L2.shape[0])
    freePtsCounts = np.ones(L2.shape[0], int) * mapRatio
    indicesRef = np.arange(0, L2.shape[0])
    for i in tqdm(range(L1.shape[0]), disable=tqdm_disable):
        minDistIndex = np.argmin(dist_matrix[i][freePtsCounts > 0])
        iMap = [L1[i], L2[freePtsCounts > 0][minDistIndex]]
        mapping.append(
            iMap[::-1] if reverse else iMap
        )
        freePtsCounts[indicesRef[freePtsCounts > 0][minDistIndex]] -= 1

    return mapping

def Mapping_1V_maxDist(Data, **params):
    '''
    Map single-vector list to single-vector list using maximum distance
    '''
    # Data
    L1, L2 = Data["1"], Data["2"]
    # Fix Data
    L1 = np.array(L1)
    L2 = np.array(L2)
    reverse = L1.shape[0] < L2.shape[0]
    if reverse: L1, L2 = L2, L1
    # Params
    tqdm_disable = "tqdm" in params.keys() and (not params["tqdm"])

    # Find Distances
    dist_matrix = params["dist"]["func"](L1, L2, **params["dist"]["params"])
    # Map
    mapping = []
    mapRatio = math.ceil(L1.shape[0] / L2.shape[0])
    freePtsCounts = np.ones(L2.shape[0], int) * mapRatio
    indicesRef = np.arange(0, L2.shape[0])
    for i in tqdm(range(L1.shape[0]), disable=tqdm_disable):
        minDistIndex = np.argmax(dist_matrix[i][freePtsCounts > 0])
        iMap = [L1[i], L2[freePtsCounts > 0][minDistIndex]]
        mapping.append(
            iMap[::-1] if reverse else iMap
        )
        freePtsCounts[indicesRef[freePtsCounts > 0][minDistIndex]] -= 1

    return mapping

def Mapping_1V_Random(Data, **params):
    '''
    Map single-vector list to single-vector list randomly
    '''
    # Data
    L1, L2 = Data["1"], Data["2"]
    # Fix Data
    L1 = np.array(L1)
    L2 = np.array(L2)
    reverse = L1.shape[0] < L2.shape[0]
    if reverse: L1, L2 = L2, L1
    # Params
    tqdm_disable = "tqdm" in params.keys() and (not params["tqdm"])

    # Map
    mapping = []
    mapRatio = math.ceil(L1.shape[0] / L2.shape[0])
    randomOrder = np.repeat(range(L2.shape[0]), mapRatio)
    np.random.shuffle(randomOrder)
    for i in tqdm(range(L1.shape[0]), disable=tqdm_disable):
        minDistIndex = randomOrder[i]
        iMap = [L1[i], L2[minDistIndex]]
        mapping.append(
            iMap[::-1] if reverse else iMap
        )

    return mapping

# Pixel Mapping (2 Vectors) - Location and Pixel Values
def Mapping_2V_minDist(Data, **params):
    '''
    Map 2-vector list to 2-vector list using minimum distance
    '''
    # Data
    L1, L2 = Data["L1"], Data["L2"]
    C1, C2 = Data["C1"], Data["C2"]
    # Fix Data
    L1 = np.array(L1)
    L2 = np.array(L2)
    C1 = np.array(C1)
    C2 = np.array(C2)
    reverse = L1.shape[0] < L2.shape[0]
    if reverse: L1, C1, L2, C2 = L2, C2, L1, C1
    # Params
    COEFF_L = params["coeffs"]["L"]
    COEFF_C = params["coeffs"]["C"]
    tqdm_disable = "tqdm" in params.keys() and (not params["tqdm"])

    # Find Distances and Combine
    dist_matrix_L = params["dist"]["L"]["func"](L1, L2, **params["dist"]["L"]["params"])
    dist_matrix_C = params["dist"]["C"]["func"](C1, C2, **params["dist"]["C"]["params"])
    dist_matrix = dist_matrix_L * COEFF_L + dist_matrix_C * COEFF_C
    # Map
    mapping_L, mapping_C = [], []
    mapRatio = math.ceil(L1.shape[0] / L2.shape[0])
    freePtsCounts = np.ones(L2.shape[0], int) * mapRatio
    indicesRef = np.arange(0, L2.shape[0])
    for i in tqdm(range(L1.shape[0]), disable=tqdm_disable):
        minDistIndex = np.argmax(dist_matrix[i][freePtsCounts > 0])
        iMap_L = [L1[i], L2[freePtsCounts > 0][minDistIndex]]
        iMap_C = [C1[i], C2[freePtsCounts > 0][minDistIndex]]
        mapping_L.append(
            iMap_L[::-1] if reverse else iMap_L
        )
        mapping_C.append(
            iMap_C[::-1] if reverse else iMap_C
        )
        freePtsCounts[indicesRef[freePtsCounts > 0][minDistIndex]] -= 1

    return mapping_L, mapping_C

def Mapping_2V_Random(Data, **params):
    '''
    Map 2-vector list to 2-vector list randomly
    '''
    # Data
    L1, L2 = Data["L1"], Data["L2"]
    C1, C2 = Data["C1"], Data["C2"]
    # Fix Data
    L1 = np.array(L1)
    L2 = np.array(L2)
    C1 = np.array(C1)
    C2 = np.array(C2)
    reverse = L1.shape[0] < L2.shape[0]
    if reverse: L1, C1, L2, C2 = L2, C2, L1, C1
    # Params
    tqdm_disable = "tqdm" in params.keys() and (not params["tqdm"])

    # Map
    mapping_L, mapping_C = [], []
    mapRatio = math.ceil(L1.shape[0] / L2.shape[0])
    randomOrder = np.repeat(range(L2.shape[0]), mapRatio)
    np.random.shuffle(randomOrder)
    for i in tqdm(range(L1.shape[0]), disable=tqdm_disable):
        minDistIndex = randomOrder[i]
        iMap_L = [L1[i], L2[minDistIndex]]
        iMap_C = [C1[i], C2[minDistIndex]]
        mapping_L.append(
            iMap_L[::-1] if reverse else iMap_L
        )
        mapping_C.append(
            iMap_C[::-1] if reverse else iMap_C
        )

    return mapping_L, mapping_C

# Main Vars
MAPPING_FUNCS = {
    "1V": {
        "Min Distance": Mapping_1V_minDist,
        "Max Distance": Mapping_1V_maxDist,
        "Random": Mapping_1V_Random
    },
    "2V": {
        "Distance": Mapping_2V_minDist,
        "Random": Mapping_2V_Random
    }
}

# Driver Code
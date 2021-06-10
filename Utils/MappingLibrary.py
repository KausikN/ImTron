'''
Library Containing many Mapping Functions
'''

# Imports
import random
import itertools
import numpy as np
from tqdm import tqdm

# Main Functions
# Location Only Mapping Functions
def Mapping_BruteForce(Data, options=None):
    L1 = Data['1']
    L2 = Data['2']
    # Check all possible mappings and take mapping with (customisable) movement
    mappings = list(itertools.permutations(range(len(L2))))
    minError = -1
    minError_Mapping = None
    for mapping in tqdm(mappings):
        Error = 0
        for i in range(len(L2)):
            Error += ((L1[i][0]-L2[mapping[i]][0])**2 + (L1[i][1]-L2[mapping[i]][1])**2)**(0.5)
        if minError == -1 or Error < minError:
            minError = Error
            minError_Mapping = mapping

    ChosenMapping = []
    for i in range(len(L2)):
        ChosenMapping.append([L1[i], L2[minError_Mapping[i]]])
        
    return ChosenMapping

def Mapping_minDist(Data, options=None):
    L1 = Data['1']
    L2 = Data['2']

    minDist_Mapping = []

    for p1 in L1:
        minDist = -1
        minDist_Point = -1
        for p2 in L2:
            dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(0.5)
            if minDist == -1 or dist < minDist:
                minDist = dist
                minDist_Point = p2.copy()
        minDist_Mapping.append([p1, minDist_Point])
        L2.remove(minDist_Point)
    return minDist_Mapping

def Mapping_maxDist(Data, options=None):
    L1 = Data['1']
    L2 = Data['2']

    maxDist_Mapping = []
    for p1 in L1:
        maxDist = -1
        maxDist_Point = -1
        for p2 in L2:
            dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(0.5)
            if maxDist == -1 or dist > maxDist:
                maxDist = dist
                maxDist_Point = p2.copy()
        maxDist_Mapping.append([p1, maxDist_Point])
        L2.remove(maxDist_Point)
    return maxDist_Mapping

# Fast Versions - Numpy Accelerated
def Mapping_minDist_Fast(Data):
    L1 = Data['1']
    L2 = Data['2']

    L1 = np.array(L1)
    L2 = np.array(L2)
    dist_matrix = np.linalg.norm(L1[:, None, :] - L2[None, :, :], axis=-1)

    minDist_Mapping = []
    takenMask = np.ones((L2.shape[0]), bool)
    indicesRef = np.arange(0, L2.shape[0])
    for i in range(L1.shape[0]):
        minDistIndex = np.argmin(dist_matrix[i][takenMask])
        minDist_Mapping.append([L1[i], L2[takenMask][minDistIndex]])
        takenMask[indicesRef[takenMask][minDistIndex]] = False

    return  minDist_Mapping

def Mapping_maxDist_Fast(Data):
    L1 = Data['1']
    L2 = Data['2']

    L1 = np.array(L1)
    L2 = np.array(L2)
    dist_matrix = np.linalg.norm(L1[:, None, :] - L2[None, :, :], axis=-1)

    maxDist_Mapping = []
    takenMask = np.ones((L2.shape[0]), bool)
    indicesRef = np.arange(0, L2.shape[0])
    for i in range(L1.shape[0]):
        maxDistIndex = np.argmax(dist_matrix[i][takenMask])
        maxDist_Mapping.append([L1[i], L2[takenMask][maxDistIndex]])
        takenMask[indicesRef[takenMask][maxDistIndex]] = False

    return  maxDist_Mapping

# Pixel Mapping Functions - Location and Pixel Values
def Mapping_LocationColorCombined(Data, options={"C_L_Ratio": 0.5, "ColorSign": 1, "LocationSign": 1, "tqdm_disable": False}):
    L1 = Data['L1']
    C1 = Data['C1']
    L2 = Data['L2']
    C2 = Data['C2']
    # Params
    C_L_Ratio = options['C_L_Ratio']
    ColorSign = options['ColorSign']
    LocationSign = options['LocationSign']
    if 'tqdm_disable' in options.keys():
        tqdm_disable = options['tqdm_disable']

    # 2 shud always have more or equal elements than 1
    swapped = False
    if len(L1) > len(L2):
        swapped = True
        L1, L2 = L2, L1
        C1, C2 = C2, C1

    # Map to closest (value and location)
    minDist_LocationMap = []
    minDist_ColorMap = []
    for i in tqdm(range(len(L1)), disable=tqdm_disable):
        p1 = L1[i]
        c1 = C1[i]
        minDist = -1
        minDist_Index = -1
        for j in range(len(L2)):
            p2 = L2[j]
            c2 = C2[j]
            locdist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(0.5)
            colordist = np.sum((np.array(c1) - np.array(c2))**2)**(0.5)
            dist = locdist*(1-C_L_Ratio)*LocationSign + C_L_Ratio*colordist*ColorSign
            if minDist == -1 or dist < minDist:
                minDist = dist
                minDist_Index = j
        if swapped:
            minDist_LocationMap.append([L2[minDist_Index], p1])
            minDist_ColorMap.append([C2[minDist_Index], c1])
        else:
            minDist_LocationMap.append([p1, L2[minDist_Index]])
            minDist_ColorMap.append([c1, C2[minDist_Index]])
        L2.pop(minDist_Index)
        C2.pop(minDist_Index)
    return minDist_LocationMap, minDist_ColorMap

def Mapping_RandomMatcher(Data, options=None):
    L1 = Data['1']
    L2 = Data['2']

    minDist_LocationMap = []

    ShuffleOrder = list(range(len(L1)))
    np.random.shuffle(ShuffleOrder)
    for i in tqdm(range(len(ShuffleOrder))):
        minDist_LocationMap.append([L1[i], L2[ShuffleOrder[i]]])

    return minDist_LocationMap

def Mapping_RandomMatcher_LocationColor(Data, options=None):
    L1 = Data['L1']
    L2 = Data['L2']
    C1 = Data['C1']
    C2 = Data['C2']

    minDist_LocationMap = []
    minDist_ColorMap = []

    ShuffleOrder = list(range(len(L1)))
    random.shuffle(ShuffleOrder)
    for i in tqdm(range(len(ShuffleOrder))):
        minDist_LocationMap.append([L1[i], L2[ShuffleOrder[i]]])
        minDist_ColorMap.append([C1[i], C2[ShuffleOrder[i]]])

    return minDist_LocationMap, minDist_ColorMap

# Fast Versions -- Numpy Accelerated
def Mapping_LocationColorCombined_Fast(Data, options={"C_L_Ratio": 0.5, "ColorSign": 1, "LocationSign": 1}):
    L1 = Data['L1']
    C1 = Data['C1']
    L2 = Data['L2']
    C2 = Data['C2']
    # Params
    C_L_Ratio = options['C_L_Ratio']
    ColorSign = options['ColorSign']
    LocationSign = options['LocationSign']

    # 2 shud always have more or equal elements than 1
    swapped = False
    if len(L1) > len(L2):
        swapped = True
        L1, L2 = L2, L1
        C1, C2 = C2, C1

    L1 = np.array(L1)
    L2 = np.array(L2)
    C1 = np.array(C1)
    C2 = np.array(C2)
    loc_dist_matrix = np.linalg.norm(L1[:, None, :] - L2[None, :, :], axis=-1)
    color_dist_matrix = np.linalg.norm(C1[:, None, :] - C2[None, :, :], axis=-1)
    dist_matrix = loc_dist_matrix*(1-C_L_Ratio)*LocationSign + C_L_Ratio*color_dist_matrix*ColorSign

    minDist_LocationMap = []
    minDist_ColorMap = []
    takenMask = np.ones((L2.shape[0]), bool)
    indicesRef = np.arange(0, L2.shape[0])
    for i in range(L1.shape[0]):
        minDistIndex = np.argmin(dist_matrix[i][takenMask])
        if swapped:
            minDist_LocationMap.append([L2[takenMask][minDistIndex], L1[i]])
            minDist_ColorMap.append([C2[takenMask][minDistIndex], C1[i]])
        else:
            minDist_LocationMap.append([L1[i], L2[takenMask][minDistIndex]])
            minDist_ColorMap.append([C1[i], C2[takenMask][minDistIndex]])
        takenMask[indicesRef[takenMask][minDistIndex]] = False

    return minDist_LocationMap, minDist_ColorMap
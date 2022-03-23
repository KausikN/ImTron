'''
Other Image Transistions
'''

# Imports
import cv2
import numpy as np

from Utils.Utils import *
from Utils.ImageSimplify import *
from Utils.ResizeFunctions import *
from Utils.MappingFunctions import *
from Utils.TransistionFunctions import *
from Utils.ImageGenerator import *

# Main Functions
# Single Pixel to Image Transistion - Color Spilt Method
def I_Transistion_SinglePixelExplode(I, StartLocation, FUNCS, N=10, **params):
    '''
    Transistion from a start location to I
    '''
    # Params
    BGColor = params["BGColor"] if "BGColor" in params.keys() else np.array([0, 0, 0])
    tqdm_disable = "tqdm" in params.keys() and (not params["tqdm"])

    # Init
    GeneratedImgs = []
    # Generate the average pixel color for StartPoint
    I_AvgColor = np.round(np.sum(np.sum(I, axis=1), axis=0) / (I.shape[0]*I.shape[1])).astype(int)
    # Initialise the starting image with one pixel coloured
    StartImg = np.ones(I.shape, dtype=int) * BGColor
    StartImg[StartLocation[0], StartLocation[1]] = I_AvgColor

    # Apply Transistion for Mappings
    # Init
    NColors_inPixel = []
    for n in range(N):
        GeneratedImgs.append(np.ones(I.shape, dtype=int) * BGColor)
        NColors_inPixel.append(np.zeros((I.shape[0], I.shape[1])))
    print("Calculating Transistion Images...")
    X = np.arange(0, I.shape[0])
    Y = np.arange(0, I.shape[1])
    Indices = np.transpose([np.tile(X, len(Y)), np.repeat(Y, len(X))])
    Indices_BGRemoved = []
    # Remove BG Pixels
    BGColorText = ",".join(BGColor.astype(str))
    for ind in Indices:
        if not (",".join(I[ind[0], ind[1]].astype(str)) == BGColorText):
            Indices_BGRemoved.append(ind)
    # Transistion
    Indices = np.array(Indices_BGRemoved)
    print("Location Count: ", Indices.shape[0])
    StartPoses = np.array([[StartLocation[0], StartLocation[1]]] * Indices.shape[0])
    X_GEN = FUNCS["transistion"]["X"]["func"](StartPoses[:, 0], Indices[:, 0], N, **FUNCS["transistion"]["X"]["params"])
    Y_GEN = FUNCS["transistion"]["Y"]["func"](StartPoses[:, 1], Indices[:, 1], N, **FUNCS["transistion"]["Y"]["params"])
    Coords = np.dstack((X_GEN, Y_GEN))
    
    # Apply
    for n in tqdm(range(N)):
        # Get Mask of Coords
        coordsMask = np.zeros(NColors_inPixel[n].shape, bool)
        maskColors = np.ones(tuple(list(NColors_inPixel[n].shape) + [3]), int)*BGColor
        coordsMask[Coords[n, :, 0], Coords[n, :, 1]] = True
        maskColors[Coords[n, :, 0], Coords[n, :, 1]] = I[Indices[:, 0], Indices[:, 1]]
        # for cd, ind in zip(Coords[n], Indices):
        #     coordsMask[cd[0], cd[1]] = True
        #     maskColors[cd[0], cd[1]] = I[ind[0], ind[1]]
        # Remove BGColor from Mask locations
        nonBGMask = (NColors_inPixel[n][:, :] == 0) & coordsMask
        GeneratedImgs[n][nonBGMask] -= BGColor
        # Add Color to Mask locations
        GeneratedImgs[n][coordsMask] += maskColors[coordsMask]
        NColors_inPixel[n][coordsMask] += 1

    # Normalize values where more than one color was added
    for n in range(N):
        mask = NColors_inPixel[n] > 0
        if FUNCS["normaliser"] == "clip":
            GeneratedImgs[n][mask] = np.clip(GeneratedImgs[n][mask], 0, 255)
        else:
            GeneratedImgs[n][mask] = (GeneratedImgs[n][mask] / np.reshape(NColors_inPixel[n][mask], (-1, 1))).astype(int)

    return GeneratedImgs, StartImg

# Driver Code
'''
Image to Image Transition based on Location
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
# Location Based Transistion
# V1 - Works only for exact pixel value matches in different locations
def I2I_Location_ExactColorMatch(I1, I2, FUNCS, N=5, **params):
    '''
    Transistion from I1 to I2 using Location Based Transistion
    '''
    # Params
    BGColor = params["BGColor"] if "BGColor" in params.keys() else np.array([0, 0, 0])
    tqdm_disable = "tqdm" in params.keys() and (not params["tqdm"])

    # Get Locations of Colours in each image
    ColoursLocations_1 = ImageColourLocations(I1, tqdm_disable)
    ColoursLocations_2 = ImageColourLocations(I2, tqdm_disable)

    # Get Location Map
    print("Calculating Location Map...")
    LocationMap = {}
    for ck in tqdm(ColoursLocations_1.keys(), disable=tqdm_disable):
        if ck in ColoursLocations_2.keys() and not ck == ",".join(BGColor.astype(str)):
            color = np.array(ck.split(","), int)
            Data = {"1": ColoursLocations_1[ck], "2": ColoursLocations_2[ck]}
            BestMapping = FUNCS["mapping"]["func"](Data, **FUNCS["mapping"]["params"])
            LocationMap[ck] = BestMapping

    # Generate Movement Transistion between Images
    # Init
    GeneratedImgs = []
    NColorsAdded_Imgs = []
    for n in range(N):
        GeneratedImgs.append(np.ones(I1.shape, int)*BGColor)
        NColorsAdded_Imgs.append(np.zeros((I1.shape[0], I1.shape[1]), int))
    # Apply Transistion for each pixel in 2 images
    print("Calculating Transistion Images...")
    for ck in tqdm(LocationMap.keys(), disable=tqdm_disable):
        Mapping = np.array(LocationMap[ck])
        color = np.array(ck.split(","), int)
        X_Mov = FUNCS["transistion"]["X"]["func"](Mapping[:, 0, 0], Mapping[:, 1, 0], N, **FUNCS["transistion"]["X"]["params"])
        Y_Mov = FUNCS["transistion"]["Y"]["func"](Mapping[:, 0, 1], Mapping[:, 1, 1], N, **FUNCS["transistion"]["Y"]["params"])
        Coords = np.dstack((X_Mov, Y_Mov))
        # Apply
        for n in range(N):
            # Get Mask of Coords
            coordsMask = np.zeros(NColorsAdded_Imgs[n].shape, bool)
            coordsMask[Coords[n, :, 0], Coords[n, :, 1]] = True
            # for c in Coords[n]:
            #     coordsMask[c[0], c[1]] = True
            # Remove BGColor from Mask locations
            nonBGMask = (NColorsAdded_Imgs[n][:, :] == 0) & coordsMask
            GeneratedImgs[n][nonBGMask] -= BGColor
            # Add Color to Mask locations
            GeneratedImgs[n][coordsMask] += color
            NColorsAdded_Imgs[n][coordsMask] += 1

    # Normalize values where more than one color was added
    for n in range(N):
        mask = NColorsAdded_Imgs[n] > 0
        if FUNCS["normaliser"] == "clip":
            GeneratedImgs[n][mask] = np.clip(GeneratedImgs[n][mask], 0, 255)
        else:
            GeneratedImgs[n][mask] = (GeneratedImgs[n][mask] / np.reshape(NColorsAdded_Imgs[n][mask], (-1, 1))).astype(int)
    
    return GeneratedImgs

# Driver Code
'''
Image to Image Transition based on Location and Color
'''

# Imports
import cv2
import pickle
import numpy as np

from Utils.Utils import *
from Utils.ImageSimplify import *
from Utils.ResizeFunctions import *
from Utils.MappingFunctions import *
from Utils.TransistionFunctions import * 
from Utils.ImageGenerator import *

# Main Functions
# V2 - Works with any 2 images - Matches Location and Color and does Transistion on both location and color
def I2I_LocationColor_Combined(I1, I2, FUNCS, N=5, **params):
    '''
    Transistion from I1 to I2 using Combined (Location and Color) Based Transistion
    '''
    # Params
    BGColors = params["BGColors"] if "BGColors" in params.keys() else [[np.array([0, 0, 0])], [np.array([0, 0, 0])]]
    loadData = params["loadData"] if "loadData" in params.keys() else False
    tqdm_disable = "tqdm" in params.keys() and (not params["tqdm"])

    if not loadData:
        # Calculate Pixel Mapping
        LocationMap, ColorMap = CalculatePixelMap(I1, I2, FUNCS["mapping"], BGColors, tqdm_disable)
        # Save Maps
        # pickle.dump(LocationMap, open(mainPath + 'LocationMap.p', 'wb'))
        # pickle.dump(ColorMap, open(mainPath + 'ColorMap.p', 'wb'))
    else:
        pass
        # Load Maps
        # LocationMap = pickle.load(open(mainPath + 'LocationMap.p', 'rb'))
        # ColorMap = pickle.load(open(mainPath + 'ColorMap.p', 'rb'))

    # Calculate Transistion Images
    GeneratedImgs = ApplyTransistionToMapping(LocationMap, ColorMap, FUNCS["transistion"], FUNCS["normaliser"], BGColors, N, I1.shape)
    
    return GeneratedImgs

def CalculatePixelMap(I1, I2, MappingFunc, BGColors=[[np.array([0, 0, 0])], [np.array([0, 0, 0])]], tqdm_disable=False):
    '''
    Calculate Pixel Mapping
    '''
    # Get Locations of Colours in each image
    ColoursLocations_1 = ImageColourLocations(I1, tqdm_disable)
    ColoursLocations_2 = ImageColourLocations(I2, tqdm_disable)

    # Init
    Locations_1 = []
    Locations_2 = []
    Colors_1 = []
    Colors_2 = []
    BGColors_str = [[",".join(c.astype(str)) for c in BGColor] for BGColor in BGColors]
    for ck in ColoursLocations_1.keys():
        color = np.array(ck.split(","), int)
        if ck not in BGColors_str[0]:
            Locations_1.extend(ColoursLocations_1[ck])
            Colors_1.extend([color] * len(ColoursLocations_1[ck]))
    for ck in ColoursLocations_2.keys():
        color = np.array(ck.split(","), int)
        if ck not in BGColors_str[1]:
            Locations_2.extend(ColoursLocations_2[ck])
            Colors_2.extend([color] * len(ColoursLocations_2[ck]))
    # Get Mapping
    print("Calculating Pixel Mapping...")
    print("Location Count: ", len(Locations_1), "-", len(Locations_2))
    LocationMap = []
    ColorMap = []
    Data = {"L1": Locations_1, "C1": Colors_1, "L2": Locations_2, "C2": Colors_2}
    LocationMap, ColorMap = MappingFunc["func"](Data, **MappingFunc["params"])

    return LocationMap, ColorMap

def ApplyTransistionToMapping(LocationMap, ColorMap, TRANSISTION_FUNCS, NORM_FUNC, BGColors, N, size):
    '''
    Apply Transistion to Mapping
    '''
    # Init
    GeneratedImgs = []
    NColorsAdded_Imgs = []
    BGColor_R_Mov = TRANSISTION_FUNCS["R"]["func"](BGColors[0][0][0], BGColors[1][0][0], N, **TRANSISTION_FUNCS["R"]["params"])
    BGColor_G_Mov = TRANSISTION_FUNCS["G"]["func"](BGColors[0][0][1], BGColors[1][0][1], N, **TRANSISTION_FUNCS["G"]["params"])
    BGColor_B_Mov = TRANSISTION_FUNCS["B"]["func"](BGColors[0][0][2], BGColors[1][0][2], N, **TRANSISTION_FUNCS["B"]["params"])
    BGColor_Mov = np.dstack((BGColor_R_Mov, BGColor_G_Mov, BGColor_B_Mov))[0]
    for n in range(N):
        GeneratedImgs.append(np.ones(size, int) * BGColor_Mov[n])
        NColorsAdded_Imgs.append(np.zeros((size[0], size[1]), int))

    # Apply Transistion for Mappings
    print("Calculating Transistion Images...")
    LocationMap = np.array(LocationMap)
    ColorMap = np.array(ColorMap)
    ColorMap_R = ColorMap[:, :, 0]
    ColorMap_G = ColorMap[:, :, 1]
    ColorMap_B = ColorMap[:, :, 2]

    X_Mov = TRANSISTION_FUNCS["X"]["func"](LocationMap[:, 0, 0], LocationMap[:, 1, 0], N, **TRANSISTION_FUNCS["X"]["params"])
    Y_Mov = TRANSISTION_FUNCS["Y"]["func"](LocationMap[:, 0, 1], LocationMap[:, 1, 1], N, **TRANSISTION_FUNCS["Y"]["params"])
    C_R_Mov = TRANSISTION_FUNCS["R"]["func"](ColorMap_R[:, 0], ColorMap_R[:, 1], N, **TRANSISTION_FUNCS["R"]["params"])
    C_G_Mov = TRANSISTION_FUNCS["G"]["func"](ColorMap_G[:, 0], ColorMap_G[:, 1], N, **TRANSISTION_FUNCS["G"]["params"])
    C_B_Mov = TRANSISTION_FUNCS["B"]["func"](ColorMap_B[:, 0], ColorMap_B[:, 1], N, **TRANSISTION_FUNCS["B"]["params"])
    
    Colors = np.dstack((C_R_Mov, C_G_Mov, C_B_Mov))
    Coords = np.dstack((X_Mov, Y_Mov))
    # Apply
    for n in range(N):
        # Get Mask of Coords
        coordsMask = np.zeros(NColorsAdded_Imgs[n].shape, bool)
        maskColors = np.ones((NColorsAdded_Imgs[n].shape[0], NColorsAdded_Imgs[n].shape[1], 3), int)*BGColor_Mov[n]
        coordsMask[Coords[n, :, 0], Coords[n, :, 1]] = True
        maskColors[Coords[n, :, 0], Coords[n, :, 1]] = Colors[n]
        # for cd, cl in zip(Coords[n], Colors[n]):
        #     coordsMask[cd[0], cd[1]] = True
        #     maskColors[cd[0], cd[1]] = cl
        # Remove BGColor from Mask locations
        nonBGMask = (NColorsAdded_Imgs[n][:, :] == 0) & coordsMask
        GeneratedImgs[n][nonBGMask] -= BGColor_Mov[n]
        # Add Color to Mask locations
        GeneratedImgs[n][coordsMask] += maskColors[coordsMask]
        NColorsAdded_Imgs[n][coordsMask] += 1

    # Normalize values where more than one color was added
    for n in range(N):
        mask = NColorsAdded_Imgs[n] > 0
        if NORM_FUNC == "clip":
            GeneratedImgs[n][mask] = np.clip(GeneratedImgs[n][mask], 0, 255)
        else:
            GeneratedImgs[n][mask] = (GeneratedImgs[n][mask] / np.reshape(NColorsAdded_Imgs[n][mask], (-1, 1))).astype(int)

    return GeneratedImgs

# Driver Code
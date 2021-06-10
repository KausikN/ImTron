'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import functools
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from Utils import Utils
from Utils import ImageSimplify
from Utils import ResizeLibrary
from Utils import MappingLibrary
from Utils import TransistionLibrary
from Utils import ImageGenerator

# Main Functions
# Location Based Transistion - 2 Images
# V1 - Works only for exact pixel value matches in different locations
def I2I_Transistion_LocationBased_ExactColorMatch(I1, I2, TransistionFuncs, MappingFunc, N=5, BGColor=np.array([0, 0, 0])):
    GeneratedImgs = []

    # Get Locations of Colours in each image
    ColoursLocations_1 = ImageColourLocations(I1)
    ColoursLocations_2 = ImageColourLocations(I2)

    # V1 - Assuming Equal No of Locations of Colors in 2 Images
    # Get the Location Map
    print("Calculating Location Map...")
    LocationMap = {}
    for ck in tqdm(ColoursLocations_1.keys()):
        if ck in ColoursLocations_2.keys() and not ck == ','.join(BGColor.astype(str)):
            color = np.array(ck.split(','), int)
            Data = {"1": ColoursLocations_1[ck], "2": ColoursLocations_2[ck]}
            BestMapping = MappingFunc(Data)
            LocationMap[ck] = BestMapping

    # Generate Movement Transistion between Images using Custom Transistion Function
    NColorsAdded_Imgs = []
    for n in range(N):
        GeneratedImgs.append(np.ones(I1.shape, int)*BGColor)
        NColorsAdded_Imgs.append(np.zeros((I1.shape[0], I1.shape[1])).astype(int))

    # Apply Transistion for each pixel in 2 images
    print("Calculating Transistion Images...")
    for ck in tqdm(LocationMap.keys()):
        Mapping = LocationMap[ck]
        color = np.array(ck.split(','), int)
        for comb in Mapping:
            # X Movement
            X_Mov = TransistionFuncs['X'](comb[0][0], comb[1][0], N)
            # Y Movement
            Y_Mov = TransistionFuncs['Y'](comb[0][1], comb[1][1], N)
            # Apply
            for n in range(N):
                if NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] == 0:
                    GeneratedImgs[n][X_Mov[n], Y_Mov[n]] = color
                    NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] = 1
                else:
                    GeneratedImgs[n][X_Mov[n], Y_Mov[n]] += color
                    NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] += 1
    for n in range(N):
        for i in range(NColorsAdded_Imgs[n].shape[0]):
            for j in range(NColorsAdded_Imgs[n].shape[1]):
                if NColorsAdded_Imgs[n][i, j] > 0:
                    GeneratedImgs[n][i, j] = GeneratedImgs[n][i, j] / NColorsAdded_Imgs[n][i, j]
    
    return GeneratedImgs


# Location Based Transistion - Numpy accelerated - Can use only Numpy TransistionFuncs
# V1 - Works only for exact pixel value matches in different locations
###################### DEV HERE - CHANGE TO FAST MAPPING AND TRANSISTION #################################
def I2I_Transistion_LocationBased_ExactColorMatch_Fast(I1, I2, TransistionFuncs, MappingFunc, N=5, BGColor=np.array([0, 0, 0])):
    GeneratedImgs = []

    # Get Locations of Colours in each image
    ColoursLocations_1 = ImageColourLocations(I1)
    ColoursLocations_2 = ImageColourLocations(I2)

    # V1 - Assuming Equal No of Locations of Colors in 2 Images
    # Get the Location Map
    print("Calculating Location Map...")
    LocationMap = {}
    for ck in tqdm(ColoursLocations_1.keys()):
        if ck in ColoursLocations_2.keys() and not ck == ','.join(BGColor.astype(str)):
            color = np.array(ck.split(','), int)
            Data = {"1": ColoursLocations_1[ck], "2": ColoursLocations_2[ck]}
            BestMapping = MappingFunc(Data)
            LocationMap[ck] = BestMapping

    # Generate Movement Transistion between Images using Custom Transistion Function
    NColorsAdded_Imgs = []
    for n in range(N):
        GeneratedImgs.append(np.ones(I1.shape, int)*BGColor)
        NColorsAdded_Imgs.append(np.zeros((I1.shape[0], I1.shape[1]), int))

    # Apply Transistion for each pixel in 2 images
    print("Calculating Transistion Images...")
    for ck in tqdm(LocationMap.keys()):
        Mapping = np.array(LocationMap[ck])
        color = np.array(ck.split(','), int)
        X_Mov = TransistionFuncs['X'](Mapping[:, 0, 0], Mapping[:, 1, 0], N)
        Y_Mov = TransistionFuncs['Y'](Mapping[:, 0, 1], Mapping[:, 1, 1], N)
        Coords = np.dstack((X_Mov, Y_Mov))
        # Apply
        for n in range(N):
            coordsMask = np.zeros(NColorsAdded_Imgs[n].shape, bool)
            for c in Coords[n]:
                coordsMask[c[0], c[1]] = True
            nonBGMask = (NColorsAdded_Imgs[n][:, :] == 0) & coordsMask
            GeneratedImgs[n][nonBGMask] -= BGColor
            GeneratedImgs[n][coordsMask] += color
            NColorsAdded_Imgs[n][coordsMask] += 1
    for n in range(N):
        mask = NColorsAdded_Imgs[n] > 0
        GeneratedImgs[n][mask] = GeneratedImgs[n][mask] / np.reshape(NColorsAdded_Imgs[n][mask], (NColorsAdded_Imgs[n][mask].shape[0], 1))
    
    return GeneratedImgs
      

# Colour Describe Function
def ImageColourLocations(I):
    ColoursLocations = {}

    if I.ndim == 2:
        I = np.reshape(I, (I.shape[0], I.shape[1], 1))
    
    for i in tqdm(range(I.shape[0])):
        for j in range(I.shape[1]):
            colourKey = ",".join(I[i, j, :].astype(str))
            if colourKey in ColoursLocations.keys():
                ColoursLocations[colourKey].append([i, j])
            else:
                ColoursLocations[colourKey] = [[i, j]]
            
    return ColoursLocations

# Driver Code
# Params
RandomImages = True
SimplifyImages = False

imgPath_1 = 'TestImgs/Test.png'
imgPath_2 = 'TestImgs/Test2.png'

imgSize = (100, 100, 3)

BGColor = [0, 0, 0]

TransistionFuncs = {
    "X": TransistionLibrary.LinearTransistion_Fast,
    "Y": TransistionLibrary.LinearTransistion_Fast
}

MappingFunc = MappingLibrary.Mapping_maxDist_Fast

ResizeFunc = functools.partial(ResizeLibrary.Resize_CustomSize, Size=imgSize)

N = 50

displayDelay = 0.0001

plotData = True
saveData = False

# Run Code
I1 = None
I2 = None

if not RandomImages:
    # Read Images
    I1 = cv2.cvtColor(cv2.imread(imgPath_1), cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(cv2.imread(imgPath_2), cv2.COLOR_BGR2RGB)

else:
    # Random Images
    # Params
    N_Colors = 10
    ColorCount_Range = (0, 100)
    Colors = list(np.random.randint(0, 255, (N_Colors, 3)))
    ColorCounts = list(np.random.randint(ColorCount_Range[0], ColorCount_Range[1], N_Colors))

    I1 = ImageGenerator.GenerateRandomImage(imgSize, BGColor, Colors, ColorCounts)
    I2 = ImageGenerator.GenerateShuffledImage(I1)
    # I2 = ImageGenerator.GenerateRandomImage(imgSize, BGColor, Colors, ColorCounts)

    # I1 = np.zeros(imgSize, int)
    # Color = [255, 255, 0]
    # I1[0, :] = Color
    # I1[-1, :] = Color
    # I1[:, 0] = Color
    # I1[:, -1] = Color
    # I2 = I1.copy()

if SimplifyImages:
    # Image Color Simplification
    # Params
    maxExtraColours = 5
    minColourDiff = 0
    DiffFunc = ImageSimplify.CheckColourCloseness_Dist_L2Norm
    DistanceFunc = ImageSimplify.EuclideanDistance

    I1 = ImageSimplify.ImageSimplify_ColorBased(I1, maxExtraColours, minColourDiff, DiffFunc, DistanceFunc)
    I2 = ImageSimplify.ImageSimplify_ColorBased(I2, maxExtraColours, minColourDiff, DiffFunc, DistanceFunc)

# Resize
I1, I2, imgSize = Utils.ResizeImages(I1, I2, ResizeFunc)

# Show Image
if plotData:
    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.show()

# Generate Transistion Images
GeneratedImgs = I2I_Transistion_LocationBased_ExactColorMatch_Fast(I1, I2, TransistionFuncs, MappingFunc, N, np.array(BGColor))

# Save
if saveData:
    saveMainPath = 'Images/'
    savePath= 'Images/LocationTrans_GIF.gif'
    mode = 'gif'
    frameSize = (imgSize[0], imgSize[1])
    fps = 60
    Utils.SaveImageSequence(GeneratedImgs, savePath, mode=mode, frameSize=None, fps=fps)

    if RandomImages:
        cv2.imwrite(saveMainPath + "LocationTrans_I1.png", I1)
        cv2.imwrite(saveMainPath + "LocationTrans_I2.png", I2)

# Display
Utils.DisplayImageSequence(GeneratedImgs, displayDelay)
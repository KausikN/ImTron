'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import pickle
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
# V2 - Works with any 2 images - Matches Location and Color and does Transistion on both location and color
def I2I_Transistion_LocationColorBased(I1, I2, TransistionFuncs_Location, TransistionFunc_Color, MappingFunc, N=5, BGColors=[[np.array([0, 0, 0])], [np.array([0, 0, 0])]], loadData=False):
    if not loadData:
        # Calculate Pixel Mapping
        LocationMap, ColorMap = CalculatePixelMap(I1, I2, MappingFunc, BGColors)

        # Save Maps
        pickle.dump(LocationMap, open(mainPath + 'LocationMap.p', 'wb'))
        pickle.dump(ColorMap, open(mainPath + 'ColorMap.p', 'wb'))
    else:
        # Load Maps
        LocationMap = pickle.load(open(mainPath + 'LocationMap.p', 'rb'))
        ColorMap = pickle.load(open(mainPath + 'ColorMap.p', 'rb'))

    # Calculate Transistion Images
    GeneratedImgs = ApplyTransistionToMapping(LocationMap, ColorMap, BGColors, TransistionFuncs_Location, TransistionFunc_Color)
    
    return GeneratedImgs

def CalculatePixelMap(I1, I2, MappingFunc, BGColors=[[np.array([0, 0, 0])], [np.array([0, 0, 0])]]):
    # Get Locations of Colours in each image
    ColoursLocations_1 = ImageColourLocations(I1)
    ColoursLocations_2 = ImageColourLocations(I2)

    Locations_1 = []
    Locations_2 = []
    Colors_1 = []
    Colors_2 = []
    for ck in ColoursLocations_1.keys():
        color = np.array(ck.split(','), int)
        if color not in BGColors[0]:
            Locations_1.extend(ColoursLocations_1[ck])
            Colors_1.extend([color]*len(ColoursLocations_1[ck]))
    for ck in ColoursLocations_2.keys():
        color = np.array(ck.split(','), int)
        if color not in BGColors[1]:
            Locations_2.extend(ColoursLocations_2[ck])
            Colors_2.extend([color]*len(ColoursLocations_2[ck]))

    # Get the Mapping
    print("Calculating Pixel Mapping...")
    print("Location Count: ", len(Locations_1), "-", len(Locations_2))
    LocationMap = []
    ColorMap = []
    Data = {"L1": Locations_1, "C1": Colors_1, "L2": Locations_2, "C2": Colors_2}
    LocationMap, ColorMap = MappingFunc(Data)

    return LocationMap, ColorMap

def ApplyTransistionToMapping(LocationMap, ColorMap, BGColors, TransistionFuncs_Location, TransistionFunc_Color):
    GeneratedImgs = []

    # Initialise Images and Vars
    Color_Movs = {}
    NColorsAdded_Imgs = []
    BGColor_Mov = TransistionFunc_Color(BGColors[0][0], BGColors[1][0], N)
    for n in range(N):
        GeneratedImgs.append(np.ones(I1.shape, int)*BGColor_Mov[n])
        NColorsAdded_Imgs.append(np.zeros((I1.shape[0], I1.shape[1])).astype(int))
    for cm in ColorMap:
        cmk = ','.join([','.join(np.array(cm[0]).astype(str)), ','.join(np.array(cm[1]).astype(str))])
        if cmk not in Color_Movs.keys():
            Color_Movs[cmk] = TransistionFunc_Color(cm[0], cm[1], N)

    # Apply Transistion for Mappings
    print("Calculating Transistion Images...")
    for lc, cc in tqdm(zip(LocationMap, ColorMap), disable=False):
        cmk = ','.join([','.join(np.array(cc[0]).astype(str)), ','.join(np.array(cc[1]).astype(str))])
        # Location Movement
        X_Mov = np.array(TransistionFuncs_Location['X'](lc[0][0], lc[1][0], N), int)
        Y_Mov = np.array(TransistionFuncs_Location['Y'](lc[0][1], lc[1][1], N), int)
        # Color Movement
        C_Mov = Color_Movs[cmk]
        # Apply
        for n in range(N):
            if NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] == 0:
                GeneratedImgs[n][X_Mov[n], Y_Mov[n]] = C_Mov[n]
                NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] = 1
            else:
                GeneratedImgs[n][X_Mov[n], Y_Mov[n]] += C_Mov[n]
                NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] += 1
    for n in range(N):
        for i in range(NColorsAdded_Imgs[n].shape[0]):
            for j in range(NColorsAdded_Imgs[n].shape[1]):
                if NColorsAdded_Imgs[n][i, j] > 0:
                    GeneratedImgs[n][i, j] = GeneratedImgs[n][i, j] / NColorsAdded_Imgs[n][i, j]
    
    return GeneratedImgs

# Fast Versions -- Numpy Accelerated
def I2I_Transistion_LocationColorBased_Fast(I1, I2, TransistionFuncs_Location, TransistionFunc_Color, MappingFunc, N=5, BGColors=[[np.array([0, 0, 0])], [np.array([0, 0, 0])]], loadData=False):
    if not loadData:
        # Calculate Pixel Mapping
        LocationMap, ColorMap = CalculatePixelMap(I1, I2, MappingFunc, BGColors)

        # Save Maps
        pickle.dump(LocationMap, open(mainPath + 'LocationMap.p', 'wb'))
        pickle.dump(ColorMap, open(mainPath + 'ColorMap.p', 'wb'))
    else:
        # Load Maps
        LocationMap = pickle.load(open(mainPath + 'LocationMap.p', 'rb'))
        ColorMap = pickle.load(open(mainPath + 'ColorMap.p', 'rb'))

    # Calculate Transistion Images
    GeneratedImgs = ApplyTransistionToMapping_Fast(LocationMap, ColorMap, BGColors, TransistionFuncs_Location, TransistionFunc_Color)
    
    return GeneratedImgs

def ApplyTransistionToMapping_Fast(LocationMap, ColorMap, BGColors, TransistionFuncs_Location, TransistionFunc_Color):
    GeneratedImgs = []

    # Initialise Images and Vars
    
    Color_Mov = []
    NColorsAdded_Imgs = []
    BGColor_Mov = TransistionFunc_Color(BGColors[0][0], BGColors[1][0], N)
    for n in range(N):
        GeneratedImgs.append(np.ones(I1.shape, int)*BGColor_Mov[n])
        NColorsAdded_Imgs.append(np.zeros((I1.shape[0], I1.shape[1]), int))

    # Apply Transistion for Mappings
    print("Calculating Transistion Images...")
    LocationMap = np.array(LocationMap)
    ColorMap = np.array(ColorMap)
    ColorMap_R = ColorMap[:, :, 0]
    ColorMap_G = ColorMap[:, :, 1]
    ColorMap_B = ColorMap[:, :, 2]

    X_Mov = TransistionFuncs_Location['X'](LocationMap[:, 0, 0], LocationMap[:, 1, 0], N)
    Y_Mov = TransistionFuncs_Location['Y'](LocationMap[:, 0, 1], LocationMap[:, 1, 1], N)
    C_R_Mov = TransistionFunc_Color(ColorMap_R[:, 0], ColorMap_R[:, 1], N)
    C_G_Mov = TransistionFunc_Color(ColorMap_G[:, 0], ColorMap_G[:, 1], N)
    C_B_Mov = TransistionFunc_Color(ColorMap_B[:, 0], ColorMap_B[:, 1], N)
    C_Mov = np.dstack((C_R_Mov, C_G_Mov, C_B_Mov))
    
    Colors = C_Mov
    Coords = np.dstack((X_Mov, Y_Mov))
    # Apply
    for n in range(N):
        coordsMask = np.zeros(NColorsAdded_Imgs[n].shape, bool)
        maskColors = np.ones(tuple(list(NColorsAdded_Imgs[n].shape) + [3]), int)*BGColor_Mov[n]
        for cd, cl in zip(Coords[n], Colors[n]):
            coordsMask[cd[0], cd[1]] = True
            maskColors[cd[0], cd[1]] = cl
        nonBGMask = (NColorsAdded_Imgs[n][:, :] == 0) & coordsMask
        GeneratedImgs[n][nonBGMask] -= BGColor_Mov[n]
        GeneratedImgs[n][coordsMask] += maskColors[coordsMask]
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
RandomImages = False
SimplifyImages = False

mainPath = 'TestImgs/'
imgPath_1 = 'TestImgs/LOL.png'
imgPath_2 = 'TestImgs/Valo_1.jpg'

imgSize = (100, 100, 3)

BGColors = [[[0, 0, 0]], [[15, 25, 35]]]
ignoreColors_N = 1

TransistionFuncs_Location = {
    'X': functools.partial(TransistionLibrary.LinearTransistion_Fast),
    'Y': functools.partial(TransistionLibrary.LinearTransistion_Fast)
}
TransistionFunc_Color = functools.partial(TransistionLibrary.LinearTransistion_Fast)

MappingFunc = functools.partial(MappingLibrary.Mapping_LocationColorCombined_Fast, options={'C_L_Ratio': 0.5, 'ColorSign': 1, 'LocationSign': 1})

ResizeFunc = functools.partial(ResizeLibrary.Resize_CustomSize, Size=imgSize)

N = 50
ImagePaddingCount = 5

displayDelay = 0.0001

plotData = True
saveData = False
loadData = False

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
    N_Colors_1 = 10
    ColorCount_Range_1 = (0, 50)
    N_Colors_2 = 5
    ColorCount_Range_2 = (0, 50)

    Colors_1 = list(np.random.randint(0, 255, (N_Colors_1, 3)))
    ColorCounts_1 = list(np.random.randint(ColorCount_Range_1[0], ColorCount_Range_1[1], N_Colors_1))

    Colors_2 = list(np.random.randint(0, 255, (N_Colors_2, 3)))
    ColorCounts_2 = list(np.random.randint(ColorCount_Range_2[0], ColorCount_Range_2[1], N_Colors_2))

    I1 = ImageGenerator.GenerateRandomImage(imgSize, BGColors[0], Colors_1, ColorCounts_1)
    I2 = ImageGenerator.GenerateRandomImage(imgSize, BGColors[1], Colors_2, ColorCounts_2)

    # I1 = np.zeros(imgSize, int)
    # Color1 = [255, 255, 0]
    # Color2 = [255, 0, 255]
    # I1[0, :] = Color1
    # I1[-1, :] = Color1
    # I1[:, 0] = Color1
    # I1[:, -1] = Color1
    # I2 = np.zeros(imgSize, int)
    # I2[0, :] = Color2
    # I2[-1, :] = Color2
    # I2[:, 0] = Color2
    # I2[:, -1] = Color2

# Resize
I1, I2, imgSize = Utils.ResizeImages(I1, I2, ResizeFunc)

if SimplifyImages:
    # Image Color Simplification
    # Params
    maxExtraColours = 5
    minColourDiff = 100
    DiffFunc = ImageSimplify.CheckColourCloseness_Dist_L2Norm
    DistanceFunc = ImageSimplify.EuclideanDistance

    TopColors = [None, None]
    I1, TopColors[0] = ImageSimplify.ImageSimplify_ColorBased(I1, maxExtraColours, minColourDiff, DiffFunc, DistanceFunc)
    I2, TopColors[1] = ImageSimplify.ImageSimplify_ColorBased(I2, maxExtraColours, minColourDiff, DiffFunc, DistanceFunc)
    BGColors[0] = TopColors[0][:ignoreColors_N]
    BGColors[1] = TopColors[1][:ignoreColors_N]

# Show Image
if plotData:
    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.show()

# Generate Transistion Images
GeneratedImgs = I2I_Transistion_LocationColorBased_Fast(I1, I2, TransistionFuncs_Location, TransistionFunc_Color, MappingFunc, N, np.array(BGColors), loadData)
# Add Padding of I1 and I2 at ends to extend duration
for i in range(ImagePaddingCount):
    GeneratedImgs.insert(0, I1)
    GeneratedImgs.append(I2)
# Save
if saveData:
    saveMainPath = 'Images/'
    saveFileName = 'LocationColorTrans_GIF.gif'
    mode = 'gif'
    frameSize = (imgSize[0], imgSize[1])
    fps = 25
    Utils.SaveImageSequence(GeneratedImgs, saveMainPath + saveFileName, mode=mode, frameSize=None, fps=fps)
    
    if RandomImages:
        cv2.imwrite(saveMainPath + "LocationColorTrans_I1.png", I1)
        cv2.imwrite(saveMainPath + "LocationColorTrans_I2.png", I2)
    else:
        cv2.imwrite(saveMainPath + "LocationColorTrans_I1.png", I1)
        cv2.imwrite(saveMainPath + "LocationColorTrans_I2.png", I2)

# Display
# if plotData:
Utils.DisplayImageSequence(GeneratedImgs, displayDelay)

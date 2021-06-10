'''
This Script allows generating a variety of other transistions
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
# Single Pixel to Image Transistion - Color Spilt Method
def I_Transistion_SinglePixelExplode(I, StartLocation, TransistionFuncs, N=10, BGColor=np.array([0, 0, 0])):
    GeneratedImgs = []

    # Generate the average pixel color for StartPoint
    I_AvgColor = np.round(np.sum(np.sum(I, axis=1), axis=0) / (I.shape[0]*I.shape[1])).astype(np.uint8)
    
    # Initialise the starting image with one pixel coloured
    StartImg = np.ones(I.shape, dtype=np.uint8) * BGColor
    StartImg[StartLocation[0], StartLocation[1]] = I_AvgColor

    # Generate the Transistion Images
    print("Calculating Transistion Images...")
    NColors_inPixel = []
    for n in range(N):
        GeneratedImgs.append(np.ones(I.shape, dtype=int) * BGColor)
        NColors_inPixel.append(np.zeros((I.shape[0], I.shape[1])))
    
    for i in tqdm(range(I.shape[0])):
        for j in range(I.shape[1]):
            X_GEN = TransistionFuncs['X'](StartLocation[0], i, N)
            Y_GEN = TransistionFuncs['Y'](StartLocation[1], j, N)
            for n in range(N):
                GeneratedImgs[n][int(X_GEN[n]), int(Y_GEN[n])] += I[i, j]
                NColors_inPixel[n][i, j] += 1
    
    # Average Colliding Pixel Colors
    for n in tqdm(range(N)):
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                GeneratedImgs[n][i, j] = (GeneratedImgs[n][i, j] / NColors_inPixel[n][i, j])
        GeneratedImgs[n] = np.array(GeneratedImgs[n], dtype=np.uint8)

    return GeneratedImgs, np.array(StartImg, dtype=np.uint8)

# Fast Version -- Numpy Accelerated
def I_Transistion_SinglePixelExplode_Fast(I, StartLocation, TransistionFuncs, N=10, BGColor=np.array([0, 0, 0])):
    GeneratedImgs = []

    BGColor = np.array(BGColor)

    # Generate the average pixel color for StartPoint
    I_AvgColor = np.round(np.sum(np.sum(I, axis=1), axis=0) / (I.shape[0]*I.shape[1])).astype(np.uint8)
    
    # Initialise the starting image with one pixel coloured
    StartImg = np.ones(I.shape, dtype=np.uint8) * BGColor
    StartImg[StartLocation[0], StartLocation[1]] = I_AvgColor

    # Generate the Transistion Images
    print("Calculating Transistion Images...")
    NColors_inPixel = []
    for n in range(N):
        GeneratedImgs.append(np.ones(I.shape, dtype=int) * BGColor)
        NColors_inPixel.append(np.zeros((I.shape[0], I.shape[1])))

    X = np.arange(0, I.shape[0])
    Y = np.arange(0, I.shape[1])
    Indices = np.transpose([np.tile(X, len(Y)), np.repeat(Y, len(X))])
    Indices_BGRemoved = []
    # Remove BG Pixels
    BGColorText = ','.join(BGColor.astype(str))
    for ind in Indices:
        if not (','.join(I[ind[0], ind[1]].astype(str)) == BGColorText):
            Indices_BGRemoved.append(ind)
    Indices_BGRemoved = np.array(Indices_BGRemoved)
    Indices = Indices_BGRemoved
    print("Location Count: ", Indices.shape[0])

    StartPoses = np.array([[StartLocation[0], StartLocation[1]]] * Indices.shape[0])
    X_GEN = TransistionFuncs['X'](StartPoses[:, 0], Indices[:, 0], N)
    Y_GEN = TransistionFuncs['Y'](StartPoses[:, 1], Indices[:, 1], N)
    Coords = np.dstack((X_GEN, Y_GEN))
    
    # Apply
    for n in tqdm(range(N)):
        coordsMask = np.zeros(NColors_inPixel[n].shape, bool)
        maskColors = np.ones(tuple(list(NColors_inPixel[n].shape) + [3]), int)*BGColor
        for cd, ind in zip(Coords[n], Indices):
            coordsMask[cd[0], cd[1]] = True
            maskColors[cd[0], cd[1]] = I[ind[0], ind[1]]
        nonBGMask = (NColors_inPixel[n][:, :] == 0) & coordsMask
        GeneratedImgs[n][nonBGMask] -= BGColor
        GeneratedImgs[n][coordsMask] += maskColors[coordsMask]
        NColors_inPixel[n][coordsMask] += 1
    
    # Average Colliding Pixel Colors
    for n in range(N):
        mask = NColors_inPixel[n] > 0
        GeneratedImgs[n][mask] = GeneratedImgs[n][mask] / np.reshape(NColors_inPixel[n][mask], (NColors_inPixel[n][mask].shape[0], 1))
        GeneratedImgs[n] = np.array(GeneratedImgs[n], dtype=np.uint8)

    return GeneratedImgs, np.array(StartImg, dtype=np.uint8)

# Driver Code
# Params
mainPath = 'TestImgs/'
imgName = 'Test2.jpg'

imgSize = (300, 300, 3)

StartLocation = (0.1, 0.1)
relativeStart = True

BGColorRange = [[0, 0, 0], [100, 100, 100]]
BGColor = [0, 0, 0]

TransistionFuncs = {
    'X': functools.partial(TransistionLibrary.LinearTransistion_Fast),
    'Y': functools.partial(TransistionLibrary.LinearTransistion_Fast)
}

divergent = True # If divergent, then GIF starts at one pixel and explodes to image -- else Image converges to one pixel

N = 50
ImagePaddingCount = 10

displayDelay = 0.0001

plotData = True
saveData = True

# Read the Image and Resize
I = cv2.imread(mainPath + imgName)
I, imgSize = Utils.ResizeImage(I, imgSize)
print(imgSize)

# Simplify BGColors
I = ImageSimplify.ImageSimplify_RangeReplace(I, valRange=BGColorRange, replaceVal=BGColor)

# Determine Start Location
if relativeStart:
    StartLocation = [int(imgSize[0] * StartLocation[0]), int(imgSize[1] * StartLocation[1])]

# Show Image
if plotData:
    plt.imshow(I)
    plt.show()

# Generate the images
GeneratedImgs, StartImg = I_Transistion_SinglePixelExplode_Fast(I, StartLocation, TransistionFuncs, N, BGColor)
# Add Original Image Padding and check for divergence and invert order if necessary
for i in range(ImagePaddingCount):
    GeneratedImgs.insert(0, StartImg)
    GeneratedImgs.append(I)
if not divergent:
    GeneratedImgs = GeneratedImgs[::-1]

# Save
if saveData:
    saveMainPath = 'Images/'

    # pickle.dump(GeneratedImgs, open(saveMainPath + 'GeneratedImgs.p', 'wb'))
    cv2.imwrite(saveMainPath + "SinglePixelExplode_I.png", cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

    saveFileName = 'SinglePixelExplode.gif'
    mode = 'gif'
    frameSize = (imgSize[0], imgSize[1])
    fps = 120
    Utils.SaveImageSequence(GeneratedImgs, saveMainPath + saveFileName, mode=mode, frameSize=None, fps=fps)
    

# Display
# if plotData:
Utils.DisplayImageSequence(GeneratedImgs, displayDelay)
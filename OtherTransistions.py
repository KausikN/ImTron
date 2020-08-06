'''
This Script allows generating a variety of other transistions
'''

# Imports
import cv2
import pickle
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
def I_Transistion_SinglePixelExplode(I, StartLocation, TransistionFunc, TransistionParams, N=10, BGColor=np.array([0, 0, 0])):
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
            L_GEN = TransistionFunc([StartLocation[0], StartLocation[1]], [i, j], N, TransistionParams)
            for n in range(N):
                GeneratedImgs[n][int(L_GEN[n][0]), int(L_GEN[n][1])] += I[i, j]
                NColors_inPixel[n][i, j] += 1
    
    # Average Colliding Pixel Colors
    for n in tqdm(range(N)):
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                GeneratedImgs[n][i, j] = (GeneratedImgs[n][i, j] / NColors_inPixel[n][i, j])
        GeneratedImgs[n] = np.array(GeneratedImgs[n], dtype=np.uint8)

    return GeneratedImgs, np.array(StartImg, dtype=np.uint8)

# Driver Code
# Params
mainPath = 'TestImgs/'
imgName = 'Test2.jpg'

imgSize = None#(100, 100, 3)

StartLocation = (2, 2)
relativeStart = True
BGColor = [0, 0, 0]

TransistionFunc = TransistionLibrary.LinearTransistion
TransistionParams = None

divergent = True # If divergent, then GIF starts at one pixel and explodes to image -- else Image converges to one pixel

N = 200
ImagePaddingCount = 10

displayDelay = 0.0001

plotData = True
saveData = True

# Read the Image and Resize
I = cv2.imread(mainPath + imgName)
I, imgSize = Utils.ResizeImage(I, imgSize)

# Determine Start Location
if relativeStart:
    StartLocation = [int(imgSize[0] / StartLocation[0]), int(imgSize[1] / StartLocation[1])]

# Show Image
if plotData:
    plt.imshow(I)
    plt.show()

# Generate the images
GeneratedImgs, StartImg = I_Transistion_SinglePixelExplode(I, StartLocation, TransistionFunc, TransistionParams, N, BGColor)
# Add Original Image Padding and check for divergence and invert order if necessary
for i in range(ImagePaddingCount):
    GeneratedImgs.insert(0, StartImg)
    GeneratedImgs.append(I)
if not divergent:
    GeneratedImgs = GeneratedImgs[::-1]

# Save
if saveData:
    saveMainPath = mainPath

    pickle.dump(GeneratedImgs, open(saveMainPath + 'GeneratedImgs.p', 'wb'))

    saveFileName = 'SinglePixelExplode.gif'
    mode = 'gif'
    frameSize = (imgSize[0], imgSize[1])
    fps = 120
    Utils.SaveImageSequence(GeneratedImgs, saveMainPath + saveFileName, mode=mode, frameSize=None, fps=fps)

# Display
# if plotData:
Utils.DisplayImageSequence(GeneratedImgs, displayDelay)
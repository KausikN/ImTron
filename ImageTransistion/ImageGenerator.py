'''
Library Containing many Image Generation Functions
'''

# Imports
import random
import numpy as np

# Main Functions
def GenerateGradient_LinearRadial(innerColor, outerColor, imgSize):
    centerPixel = (int(imgSize[0]/2), int(imgSize[1]/2))

    I = np.zeros(imgSize).astype(np.uint8)
    I[centerPixel[0], centerPixel[1]] = innerColor
    I[-1, -1] = outerColor # Outer most pixel in any case of size is final pixel
    maxDist = ((imgSize[0]-centerPixel[0])**2 + (imgSize[1]-centerPixel[1])**2)**(0.5)

    # Color Images
    if len(imgSize) <= 2:
        for i in range(imgSize[0]):
            for j in range(imgSize[1]):
                dist = ((i-centerPixel[0])**2 + (j-centerPixel[1])**2)**(0.5)
                fracVal = dist / maxDist
                I[i, j] = int(outerColor * fracVal + innerColor * (1-fracVal))
    # Grayscale Images
    else:
        for i in range(imgSize[0]):
            for j in range(imgSize[1]):
                dist = ((i-centerPixel[0])**2 + (j-centerPixel[1])**2)**(0.5)
                fracVal = dist / maxDist
                I[i, j] = list((outerColor * fracVal + innerColor * (1-fracVal)).astype(np.uint8))

    return I


def GenerateRandomImage(imgSize, BGColor, Colors, ColorCounts):
    I = np.ones(imgSize, int)*BGColor
    totalPixelCount = imgSize[0]*imgSize[1]
    colorPixelsCount = sum(ColorCounts)
    BGColorCount = totalPixelCount - colorPixelsCount
    if BGColorCount >= 0:
        order = np.array([-1]*totalPixelCount)
        curIndex = 0
        for i in range(len(ColorCounts)):
            order[curIndex : curIndex + ColorCounts[i]] = i
            curIndex += ColorCounts[i]
        random.shuffle(order)
    I_Colors = np.reshape(np.array(order), (imgSize[0], imgSize[1]))
    for i in range(I_Colors.shape[0]):
        for j in range(I_Colors.shape[1]):
            if not I_Colors[i, j] == -1:
                I[i, j] = Colors[I_Colors[i, j]]
    return I
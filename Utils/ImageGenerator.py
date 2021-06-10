'''
Library Containing many Image Generation Functions
'''

# Imports
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from Utils import GradientLibrary

# Main Functions
# Gradient Images
# Linear Gradients
def GenerateGradient_LinearLinear(StartColor, EndColor, imgSize, Rotation=0, gradient_n=200):
    Start_HEX = GradientLibrary.RGB2Hex(StartColor)
    End_HEX = GradientLibrary.RGB2Hex(EndColor)
    I = GradientLibrary.LineGradient2Image(GradientLibrary.Gradient(GradientLibrary.LineGradient_Linear(Start_HEX, End_HEX, gradient_n), Rotation), imgSize)
    return I

# Radial Gradients
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

# Random Images
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

# Shuffled Image
def GenerateShuffledImage(I):
    if I.ndim <= 2:
        I = np.reshape(I, (I.shape[0], I.shape[1], 1))
    I_flat = np.reshape(I, (I.shape[0]*I.shape[1], I.shape[2]))
    I_shuffled_flat = np.copy(I_flat)
    np.random.shuffle(I_shuffled_flat)
    I_shuffled = np.reshape(I_shuffled_flat, (I.shape[0], I.shape[1], I.shape[2]))
    return I_shuffled

# Text Images
def GenerateTextImages(text, imgSize, coord, font=3, fontScale=1, fontColor=[0, 0, 0], BGColor=[255, 255, 255], thickness=1):
    I = np.ones(imgSize, int) * np.array(BGColor)
    I = cv2.putText(I, text, coord, font, fontScale, tuple(fontColor), thickness)
    return I

'''
# Driver Code
imgSize = (100, 100, 3)
text = "K"
coord = (50, 50)
fontColor = [0, 0, 0]
BGColor = [255, 255, 255]
fontScale = 1
thickness = 1
font = 3
I = GenerateTextImages(text, imgSize, coord, font=font, fontScale=fontScale, fontColor=fontColor, BGColor=BGColor, thickness=thickness)
print(I.ndim)
plt.imshow(I)
plt.show()
'''
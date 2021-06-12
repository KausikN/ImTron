'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from tqdm import tqdm

from Utils import Utils
from Utils import ResizeLibrary
from Utils import TransistionLibrary
from Utils import ImageGenerator

# Main Functions
# Colour Based Gradient Transistion - 2 Images
def I2I_Transistion_ColorGradient(I1, I2, TransistionFunc, N=5):
    GeneratedImgs = []

    for n in range(N):
        GeneratedImgs.append(np.zeros(I1.shape).astype(np.uint8))

    # Apply Transistion for each pixel in 2 images
    for i in tqdm(range(I1.shape[0])):
        for j in range(I1.shape[1]):
            GeneratedPixels = TransistionFunc(I1[i, j], I2[i, j], N)
            for n in range(N):
                GeneratedImgs[n][i, j] = list(GeneratedPixels[n])

    return GeneratedImgs

# Colour Based Gradient Transistion - Numpy accelerated - Can use only Numpy TransistionFuncs
def I2I_Transistion_ColorGradient_Fast(I1, I2, TransistionFunc, N=5):
    GeneratedImgs = []

    # Apply Transistion for all pixels in 2 images
    GeneratedImgs = TransistionFunc(I1, I2, N)
    GeneratedImgs = list(GeneratedImgs)

    return GeneratedImgs


# # Driver Code
# # Params
# RandomImages = True

# imgPath_1 = 'TestImgs/Test.jpg'
# imgPath_2 = 'TestImgs/Test2.jpg'

# imgSize = (300, 300, 3)

# TransistionFunc = TransistionLibrary.LinearTransistion_Fast

# ResizeFunc = functools.partial(ResizeLibrary.Resize_MaxSize)

# N = 5

# displayDelay = 0.01

# plotData = True
# saveData = False

# # Run Code
# I1 = None
# I2 = None

# if not RandomImages:
#     # Read Images
#     I1 = cv2.cvtColor(cv2.imread(imgPath_1), cv2.COLOR_BGR2RGB)
#     I2 = cv2.cvtColor(cv2.imread(imgPath_2), cv2.COLOR_BGR2RGB)
# else:
#     # Random Images
#     I1 = ImageGenerator.GenerateGradient_LinearRadial(np.array([255, 255, 255]), np.array([255, 0, 0]), imgSize)
#     I2 = ImageGenerator.GenerateGradient_LinearRadial(np.array([0, 0, 255]), np.array([255, 255, 255]), imgSize)

# # Resize
# I1, I2, imgSize = Utils.ResizeImages(I1, I2, ResizeFunc)

# # Display
# if plotData:
#     plt.subplot(1, 2, 1)
#     plt.imshow(I1)
#     plt.subplot(1, 2, 2)
#     plt.imshow(I2)
#     plt.show()

# # Generate Transistion Images
# GeneratedImgs = I2I_Transistion_ColorGradient_Fast(I1, I2, TransistionFunc, N)
# # Loop Back to 1st image
# GeneratedImgs.extend(GeneratedImgs[::-1])

# # Save
# if saveData:
#     saveMainPath = 'TestImgs/'
#     saveFileName = 'ColorTrans.gif'
#     mode = 'gif'
#     frameSize = (imgSize[0], imgSize[1])
#     fps = 25
#     Utils.SaveImageSequence(GeneratedImgs, saveMainPath + saveFileName, mode=mode, frameSize=None, fps=fps)
    
#     if RandomImages:
#         cv2.imwrite(saveMainPath + "ColorTrans_I1.png", I1)
#         cv2.imwrite(saveMainPath + "ColorTrans_I2.png", I2)

# # Display
# Utils.DisplayImageSequence(GeneratedImgs, displayDelay)
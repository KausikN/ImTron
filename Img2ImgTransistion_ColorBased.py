'''
Image to Image Transition based on Color
'''

# Imports
import cv2
import numpy as np

from Utils.Utils import *
from Utils.ResizeFunctions import *
from Utils.TransistionFunctions import *
from Utils.ImageGenerator import *

# Main Functionss
# Colour Based Gradient Transistion
def I2I_Color_ExactLocationMatch(I1, I2, FUNCS, N=5, **params):
    '''
    Generate a transistion from I1 to I2 using a colour gradient
    '''
    # Apply Transistion for Mappings
    print("Calculating Transistion Images...")
    ColorMap_R = np.dstack((I1[:, :, 0], I2[:, :, 0]))
    ColorMap_G = np.dstack((I1[:, :, 1], I2[:, :, 1]))
    ColorMap_B = np.dstack((I1[:, :, 2], I2[:, :, 2]))

    C_R_Mov = FUNCS["transistion"]["R"]["func"](ColorMap_R[:, :, 0], ColorMap_R[:, :, 1], N, **FUNCS["transistion"]["R"]["params"])
    C_G_Mov = FUNCS["transistion"]["G"]["func"](ColorMap_G[:, :, 0], ColorMap_G[:, :, 1], N, **FUNCS["transistion"]["G"]["params"])
    C_B_Mov = FUNCS["transistion"]["B"]["func"](ColorMap_B[:, :, 0], ColorMap_B[:, :, 1], N, **FUNCS["transistion"]["B"]["params"])

    # Generate Transistion Images
    GeneratedImgs = []
    for n in range(N):
        I = np.zeros((I1.shape[0], I1.shape[1], 3), np.uint8)
        I[:, :, 0] = C_R_Mov[n, :, :]
        I[:, :, 1] = C_G_Mov[n, :, :]
        I[:, :, 2] = C_B_Mov[n, :, :]
        GeneratedImgs.append(I)

    return GeneratedImgs

# Driver Code
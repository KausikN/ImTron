'''
Gradient Colour Generation and Viewing Functions
'''

# Imports
import cv2
import numpy as np
from numpy import random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Classes
class Gradient:
    def __init__(self, color_list, angle):
        self.color_list = color_list
        self.angle = angle

# Utils Functions
def LinearGradient_Library(start_color, end_color, n):
    '''
    Generate a library of linear gradients
    '''
    return list(start_color.range_to(end_color, n))

def Hex2RGB(color):
    '''
    Convert HEX to RGB
    '''
    return list(mpl.colors.to_rgb(color))

def RGB2Hex(color):
    '''
    Convert RGB to HEX
    '''
    return mpl.colors.to_hex(color)

def RotateImage(I, angle, filled=False, tqdmDisable=True):
    '''
    Rotate image
    '''
    image_center = tuple(np.array(I.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(I, rot_mat, I.shape[1::-1], flags=cv2.INTER_LINEAR)

    if filled:
        box = [[], []]
        for i in tqdm(range(result.shape[0]), disable=tqdmDisable):
            for j in range(result.shape[1]):
                # Top Left
                if len(box[0]) == 0 and np.sum(result[i, j]) > 0:
                    box[0] = (i, j)
                # Bottom Right
                if len(box[1]) == 0 and np.sum(result[-i-1, -j-1]) > 0:
                    box[1] = (result.shape[0] - i - 1, result.shape[1] - j - 1)
        BoxImg = result[box[0][0]:box[1][0], box[0][1]:box[1][1]]
        result = cv2.resize(BoxImg, (I.shape[0], I.shape[1]))

    return result

def ColorMix(c1, c2, mix=0): 
    '''
    Mix two colors
    '''
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

# Main Functions
# Line Gradient Functions
def LineGradient_Linear(start_color, end_color, n, tqdmDisable=True):
    '''
    Generate a linear line gradient
    '''
    gradient = []
    for x in tqdm(range(n+1), disable=tqdmDisable):
        gradient.append(ColorMix(start_color, end_color, x/n))
    return gradient

def LineGradient_Display(gradient, linewidth=4, figsize=(8, 5), tqdmDisable=True):
    '''
    Display a line gradient
    '''
    fig, ax = plt.subplots(figsize=figsize)
    for gi in tqdm(range(len(gradient)), disable=tqdmDisable):
        ax.axvline(gi+1, color=gradient[gi], linewidth=linewidth) 
    plt.show()

def LineGradient2Image(gradient, imgsize, filled=True):
    '''
    Generate an image from a line gradient
    '''
    # Convert to rgb
    for i in range(len(gradient.color_list)):
        gradient.color_list[i] = list(mpl.colors.to_rgb(gradient.color_list[i]))

    I = np.array([gradient.color_list]*len(gradient.color_list))
    I = cv2.resize(I, imgsize)

    I = RotateImage(I, gradient.angle, filled=filled)

    return I

def LineGradients_Add(g1, g2, imgsize):
    '''
    Add two line gradients
    '''
    I1 = LineGradient2Image(g1, imgsize)
    I2 = LineGradient2Image(g2, imgsize)

    return I1 + I2

# Driver Code
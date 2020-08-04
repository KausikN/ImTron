'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import random
import imageio
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from tqdm import tqdm

# Main Functions
def ResizeImages(I1, I2, ResizeFunc=None, ResizeParams=None):
    print("Before Resizing: I1:", I1.shape, "I2:", I2.shape)
    # Resize Match the 2 images - Final I1 and I2 must be of same size
    if not ResizeFunc == None:
        if ResizeParams == None:
            I1, I2 = ResizeFunc(I1, I2)
        else:
            I1, I2 = ResizeFunc(I1, I2, ResizeParams)
    print("After Resizing: I1:", I1.shape, "I2:", I2.shape)
    return I1, I2, I1.shape

def DisplayImageSequence(ImgSeq, delay=1):
    imgIndex = 0
    N = len(ImgSeq)
    while(True):
        plt.figure(1)
        plt.clf()
        plt.imshow(ImgSeq[imgIndex])
        plt.title(str(imgIndex+1))

        plt.pause(delay)
        imgIndex = (imgIndex + 1) % N

def SaveImageSequence(ImgSeq, path, mode='gif', frameSize=None, fps=25):
    # modes
    # gif
    if mode in ['gif', 'GIF', 'G', 'g']:
        imageio.mimsave(path, ImgSeq)
    # Video
    elif mode in ['V', 'v', 'Video', 'video', 'VIDEO', 'VID', 'vid']:
        if frameSize == None:
            frameSize = (ImgSeq[0].shape[0], ImgSeq[0].shape[1])
        vid = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
        for i in range(len(ImgSeq)):
            vid.write(ImgSeq[i])
        vid.release()
    # Images
    else:
        for i in range(len(ImgSeq)):
            cv2.imwrite(path + str(i+1), ImgSeq[i])
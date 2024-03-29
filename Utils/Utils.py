'''
Utils Functions
'''

# Imports
import os
import cv2
import imageio
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
def Grayscale2ColorFormat(I):
    '''
    Convert Grayscale Image to Color Format
    '''
    if I.ndim == 2:
        I = np.reshape(I, (I.shape[0], I.shape[1], 1))
    return I

def ResizeImage(I, Size):
    '''
    Resize Image to Size
    '''
    I = cv2.resize(I, (Size[0], Size[1]))
    return I, I.shape

def ResizeImages(I1, I2, ResizeFunc=None, ResizeParams={}):
    '''
    Resize Images to Same Size
    '''
    # Resize Match the 2 images - Final I1 and I2 must be of same size
    print("Before Resizing: I1:", I1.shape, "I2:", I2.shape)
    I1, I2 = ResizeFunc(I1, I2, **ResizeParams)
    print("After Resizing: I1:", I1.shape, "I2:", I2.shape)
    return I1, I2, I1.shape

def ImageColourLocations(I, tqdm_disable=False):
    '''
    Get the locations of all colours in an image
    '''
    ColoursLocations = {}
    # Init Colors
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            colourKey = ",".join(I[i, j, :].astype(str))
            ColoursLocations[colourKey] = []    
    # Get Locations
    for i in tqdm(range(I.shape[0]), disable=tqdm_disable):
        for j in range(I.shape[1]):
            colourKey = ",".join(I[i, j, :].astype(str))
            ColoursLocations[colourKey].append([i, j])
            
    return ColoursLocations

def DisplayImageSequence(ImgSeq, delay=1):
    '''
    Display Image Sequence
    '''
    imgIndex = 0
    N = len(ImgSeq)
    while(True):
        plt.figure(1)
        plt.clf()
        plt.imshow(ImgSeq[imgIndex])
        plt.title(str(imgIndex+1))

        plt.pause(delay)
        imgIndex = (imgIndex + 1) % N

# OLD Function
def SaveImageSequence(ImgSeq, path, mode='gif', frameSize=None, fps=25):
    '''
    Save Image Sequence - OLD
    '''
    # modes
    # gif
    if mode.lower() in ['gif', 'g']:
        imageio.mimsave(path, ImgSeq)
    # Video
    elif mode.lower() in ['v', 'video', 'vid']:
        if frameSize == None:
            frameSize = (ImgSeq[0].shape[0], ImgSeq[0].shape[1])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid = cv2.VideoWriter(path, fourcc, fps, frameSize)
        for i in range(len(ImgSeq)):
            vid.write(ImgSeq[i])
        vid.release()
    # Images
    else:
        for i in range(len(ImgSeq)):
            cv2.imwrite(path + str(i+1), ImgSeq[i])

# NEW / Updated Function
def SaveFrames2Video(frames, pathOut, fps=20.0, size=None):
    '''
    Save Frames to Video
    '''
    if os.path.splitext(pathOut)[-1] == '.gif':
        frames_images = [Image.fromarray(frame) for frame in frames]
        extraFrames = []
        if len(frames_images) > 1:
            extraFrames = frames_images[1:]
        frames_images[0].save(pathOut, save_all=True, append_images=extraFrames, format='GIF', loop=0)
    else:
        if size is None: size = (frames[0].shape[1], frames[0].shape[0])
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
        for frame in frames:
            out.write(frame)
        out.release()

def FixVideoFile(pathIn, pathOut):
    '''
    Fix Video File for displaying in streamlit
    '''
    COMMAND_VIDEO_CONVERT = 'ffmpeg -i \"{path_in}\" -vcodec libx264 \"{path_out}\"'
    
    if os.path.exists(pathOut):
        os.remove(pathOut)

    convert_cmd = COMMAND_VIDEO_CONVERT.format(path_in=pathIn, path_out=pathOut)
    print("Running Conversion Command:")
    print(convert_cmd + "\n")
    ConvertOutput = subprocess.getoutput(convert_cmd)

# Driver Code
'''
Image Resize Functions
'''

# Imports
import cv2
import numpy as np

# Main Functions
def Resize_CustomSize(I1, I2, **params):
    '''
    Resize Images to Custom Size
    '''
    # Params
    size = params["size"]
    # Resize
    if I1 is not None and not size == (I1.shape[0], I1.shape[1]):
        I1 = cv2.resize(I1, (size[0], size[1]))
    if I2 is not None and not size == (I2.shape[0], I2.shape[1]):
        I2 = cv2.resize(I2, (size[0], size[1]))

    return I1, I2

def Resize_MaxSize(I1, I2, **params):
    '''
    Resize Images to Max Size of Both Images
    '''
    # Resize
    CommonSize = (max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1]))
    if not CommonSize == (I1.shape[0], I1.shape[1]):
        I1 = cv2.resize(I1, CommonSize)
    if not CommonSize == (I2.shape[0], I2.shape[1]):
        I2 = cv2.resize(I2, CommonSize)

    return I1, I2

def Resize_MinSize(I1, I2, **params):
    '''
    Resize Images to Min Size of Both Images
    '''
    # Resize
    CommonSize = (min(I1.shape[0], I2.shape[0]), min(I1.shape[1], I2.shape[1]))
    if not CommonSize == (I1.shape[0], I1.shape[1]):
        I1 = cv2.resize(I1, CommonSize)
    if not CommonSize == (I2.shape[0], I2.shape[1]):
        I2 = cv2.resize(I2, CommonSize)

    return I1, I2

def Resize_PaddingFillMaxSize(I1, I2, **params):
    '''
    Resize Images to Max Size of Both Images using padding
    '''
    # Find Max Size and create new images
    CommonSize = (max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1]), I1.shape[2])
    I1_R = np.zeros(CommonSize).astype(int)
    I2_R = np.zeros(CommonSize).astype(int)
    # Fill Image Parts and align to center
    # I1
    PaddingSize = (CommonSize[0] - I1.shape[0], CommonSize[1] - I1.shape[1])
    I_start = (int(PaddingSize[0]/2), int(PaddingSize[1]/2))
    I_end = (I_start[0] + I1.shape[0], I_start[1] + I1.shape[1])
    I1_R[I_start[0]:I_end[0], I_start[1]:I_end[1]] = I1
    # I2
    PaddingSize = (CommonSize[0] - I2.shape[0], CommonSize[1] - I2.shape[1])
    I_start = (int(PaddingSize[0]/2), int(PaddingSize[1]/2))
    I_end = (I_start[0] + I2.shape[0], I_start[1] + I2.shape[1])
    I2_R[I_start[0]:I_end[0], I_start[1]:I_end[1]] = I2

    return I1_R, I2_R

def Resize_CropMinSize(I1, I2, **params):
    '''
    Resize Images to Min Size of Both Images using cropping
    '''
    # Find Min Size and create new images
    CommonSize = (min(I1.shape[0], I2.shape[0]), min(I1.shape[1], I2.shape[1]), I1.shape[2])
    I1_R = np.zeros(CommonSize).astype(int)
    I2_R = np.zeros(CommonSize).astype(int)
    # Crop Image Parts and align to center
    # I1
    CropSize = (I1.shape[0] - CommonSize[0], I1.shape[1] - CommonSize[1])
    I_start = (int(CropSize[0]/2), int(CropSize[1]/2))
    I_end = (I_start[0] + CommonSize[0], I_start[1] + CommonSize[1])
    I1_R = I1[I_start[0]:I_end[0], I_start[1]:I_end[1]]
    # I2
    CropSize = (I2.shape[0] - CommonSize[0], I2.shape[1] - CommonSize[1])
    I_start = (int(CropSize[0]/2), int(CropSize[1]/2))
    I_end = (I_start[0] + CommonSize[0], I_start[1] + CommonSize[1])
    I2_R = I2[I_start[0]:I_end[0], I_start[1]:I_end[1]]

    return I1_R, I2_R

# Main Vars
RESIZE_FUNCS = {
    "Custom Size": Resize_CustomSize,
    "Max Size": Resize_MaxSize,
    "Min Size": Resize_MinSize,
    "Padding Fill Max Size": Resize_PaddingFillMaxSize,
    "Crop to Min Size": Resize_CropMinSize
}

# Driver Code
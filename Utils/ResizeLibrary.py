'''
Library Containing many Resize Functions
'''

# Imports
import cv2
import numpy as np

# Main Functions
def Resize_CustomSize(I1, I2, Size):
    if not Size == (I1.shape[0], I1.shape[1]):
        I1 = cv2.resize(I1, (Size[0], Size[1]))
    if not Size == (I2.shape[0], I2.shape[1]):
        I2 = cv2.resize(I2, (Size[0], Size[1]))
    return I1, I2

def Resize_MaxSize(I1, I2):
    CommonSize = (max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1]))
    if not CommonSize == (I1.shape[0], I1.shape[1]):
        I1 = cv2.resize(I1, CommonSize)
    if not CommonSize == (I2.shape[0], I2.shape[1]):
        I2 = cv2.resize(I2, CommonSize)
    return I1, I2

def Resize_PaddingFillMaxSize(I1, I2):
    # Colour Images
    if I1.ndim == 3:
        # Find Max Size and create new images
        CommonSize = (max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1]), I1.shape[2])
    else:
        # Find Max Size and create new images
        CommonSize = (max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1]))

    I1_R = np.zeros(CommonSize).astype(int)
    I2_R = np.zeros(CommonSize).astype(int)
    # Fill Image Parts and align to center
    # I1
    PaddingSize = (CommonSize[0] - I1.shape[0], CommonSize[1] - I1.shape[1])
    ImgPart_Start = (int(PaddingSize[0]/2), int(PaddingSize[1]/2))
    ImgPart_End = (ImgPart_Start[0] + I1.shape[0], ImgPart_Start[1] + I1.shape[1])
    I1_R[ImgPart_Start[0]:ImgPart_End[0], ImgPart_Start[1]:ImgPart_End[1]] = I1
    # I2
    PaddingSize = (CommonSize[0] - I2.shape[0], CommonSize[1] - I2.shape[1])
    ImgPart_Start = (int(PaddingSize[0]/2), int(PaddingSize[1]/2))
    ImgPart_End = (ImgPart_Start[0] + I2.shape[0], ImgPart_Start[1] + I2.shape[1])
    I2_R[ImgPart_Start[0]:ImgPart_End[0], ImgPart_Start[1]:ImgPart_End[1]] = I2

    return I1_R, I2_R
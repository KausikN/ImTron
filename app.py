"""
Stream lit GUI for hosting ImageColoriser
"""

# Imports
import cv2
import functools
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json

import Img2ImgTransistion_ColorBased
import Img2ImgTransistion_LocationBased
import Img2ImgTransistion_LocationColorBased

from Utils import MappingLibrary
from Utils import TransistionLibrary
from Utils import ResizeLibrary
from Utils import ImageSimplify
from Utils import ImageGenerator

# Main Vars
config = json.load(open('./StreamLitGUI/UIConfig.json', 'r'))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
        tuple(
            [config['PROJECT_NAME']] + 
            config['PROJECT_MODES']
        )
    )
    
    if selected_box == config['PROJECT_NAME']:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(' ', '_').lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config['PROJECT_NAME'])
    st.markdown('Github Repo: ' + "[" + config['PROJECT_LINK'] + "](" + config['PROJECT_LINK'] + ")")
    st.markdown(config['PROJECT_DESC'])

    # st.write(open(config['PROJECT_README'], 'r').read())

#############################################################################################################################
# Repo Based Vars
DEFAULT_PATH_EXAMPLEIMAGE1 = 'TestImgs/LS_1.jpg'
DEFAULT_PATH_EXAMPLEIMAGE2 = 'TestImgs/LS_2.jpg'

DEFAULT_SAVEPATH = 'TestImgs/RandomImage.png'

TRANSISTIONFUNCS = {
    "Linear": TransistionLibrary.LinearTransistion_Fast
}
RESIZEFUNCS = {
    "Custom Size": ResizeLibrary.Resize_CustomSize,
    "Minimum Size": ResizeLibrary.Resize_MinSize,
    "Maximum Size": ResizeLibrary.Resize_MaxSize,
    "Padded Maximum Size": ResizeLibrary.Resize_PaddingFillMaxSize,
}
MAPPINGFUNCS_LOCATION = {
    "Minimum Distance": MappingLibrary.Mapping_minDist_Fast,
    "Maximum Distance": MappingLibrary.Mapping_maxDist_Fast,
    "Random": MappingLibrary.Mapping_RandomMatcher
}
MAPPINGFUNCS_COLORLOCATION = {
    "Distance Based": MappingLibrary.Mapping_LocationColorCombined_Fast,
    "Random": MappingLibrary.Mapping_RandomMatcher_LocationColor
}

IMAGESIZE_MIN = [1, 1]
IMAGESIZE_MAX = [512, 512]
IMAGESIZE_DEFAULT = [100, 100]
IMAGESIZEINDICATORIMAGE_SIZE = [128, 128]

DISPLAY_DELAY = 0.1

# Util Vars


# Util Functions
def Hex_to_RGB(val):
    val = val.lstrip('#')
    lv = len(val)
    return tuple(int(val[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def RGB_to_Hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

@st.cache
def GenerateImageSizeIndicatorImage(ImageSize):
    ### Image Size Indicator Image 
    ImageSizeIndicator_Image = np.zeros((IMAGESIZEINDICATORIMAGE_SIZE[0], IMAGESIZEINDICATORIMAGE_SIZE[1]), dtype=int)
    ImageSizeIndicator_Image[:int((ImageSize[0]/IMAGESIZE_MAX[0])*IMAGESIZEINDICATORIMAGE_SIZE[0]), :int((ImageSize[1]/IMAGESIZE_MAX[1])*IMAGESIZEINDICATORIMAGE_SIZE[1])] = 255
    return ImageSizeIndicator_Image

# @st.cache
def GenerateRandomImage(ImageSize, USERINPUT_BGColor, USERINPUT_NColors, ColorCount_Range):
    Colors = list(np.random.randint(0, 255, (USERINPUT_NColors, 3)))
    ColorCounts = list(np.random.randint(ColorCount_Range[0], ColorCount_Range[1], USERINPUT_NColors))
    USERINPUT_Image_1 = ImageGenerator.GenerateRandomImage(ImageSize, USERINPUT_BGColor, Colors, ColorCounts)
    return USERINPUT_Image_1

@st.cache
def ShuffleImage_Cached(I):
    I_Shuffled = ImageGenerator.GenerateShuffledImage(I)
    return I_Shuffled

# Main Functions
@st.cache
def ColorBasedTransistion(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFunc, N):
    GeneratedImgs = Img2ImgTransistion_ColorBased.I2I_Transistion_ColorGradient_Fast(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFunc, N)
    return GeneratedImgs

@st.cache
def LocationBasedTransistion(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFunc_X, TransistionFunc_Y, MappingFunc, N, BGColor):
    TransistionFuncs = {
        'X': TransistionFunc_X,
        'Y': TransistionFunc_Y
    }
    GeneratedImgs = Img2ImgTransistion_LocationBased.I2I_Transistion_LocationBased_ExactColorMatch_Fast(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFuncs, MappingFunc, N, BGColor)
    return GeneratedImgs

# UI Functions
def UI_DisplayImageSequence(GeneratedImgs):
    outputDisplay = st.empty()
    while(True):
        for i in range(len(GeneratedImgs)):
            plt.pause(DISPLAY_DELAY)
            outputDisplay.image(GeneratedImgs[i], caption='Generated Transistion', use_column_width=True)

def UI_LoadImageFiles():
    USERINPUT_SwapImages = st.checkbox("Swap Images")
    col1, col2 = st.beta_columns(2)

    USERINPUT_ImageData_1 = col1.file_uploader("Upload Start Image", ['png', 'jpg', 'jpeg', 'bmp'])
    if USERINPUT_ImageData_1 is not None:
        USERINPUT_ImageData_1 = USERINPUT_ImageData_1.read()
    else:
        USERINPUT_ImageData_1 = open(DEFAULT_PATH_EXAMPLEIMAGE1, 'rb').read()
    USERINPUT_Image_1 = cv2.imdecode(np.frombuffer(USERINPUT_ImageData_1, np.uint8), cv2.IMREAD_COLOR)
    USERINPUT_Image_1 = cv2.cvtColor(USERINPUT_Image_1, cv2.COLOR_BGR2RGB)
    
    USERINPUT_ImageData_2 = col2.file_uploader("Upload End Image", ['png', 'jpg', 'jpeg', 'bmp'])
    if USERINPUT_ImageData_2 is not None:
        USERINPUT_ImageData_2 = USERINPUT_ImageData_2.read()
    else:
        USERINPUT_ImageData_2 = open(DEFAULT_PATH_EXAMPLEIMAGE2, 'rb').read()
    USERINPUT_Image_2 = cv2.imdecode(np.frombuffer(USERINPUT_ImageData_2, np.uint8), cv2.IMREAD_COLOR)
    USERINPUT_Image_2 = cv2.cvtColor(USERINPUT_Image_2, cv2.COLOR_BGR2RGB)

    if USERINPUT_SwapImages:
        USERINPUT_Image_1, USERINPUT_Image_2 = USERINPUT_Image_2, USERINPUT_Image_1

    col1.image(USERINPUT_Image_1, caption="Start Image", use_column_width=True)
    col2.image(USERINPUT_Image_2, caption="End Image", use_column_width=True)

    return USERINPUT_Image_1, USERINPUT_Image_2

def UI_LoadGenerateImages():
    global REGEN_KEY

    st.markdown("###Generate Images")

    USERINPUT_Reshuffle = False
    USERINPUT_Regenerate = False

    USERINPUT_SwapImages = st.checkbox("Swap Images")
    USERINPUT_BGColor = []

    GENERATE_METHODS = ["Upload Image", "Generate Random Image"]
    USERINPUT_GenerateMethod = st.selectbox("Select Generate Method", GENERATE_METHODS)
    col1, col2 = st.beta_columns(2)
    # Upload
    if USERINPUT_GenerateMethod == GENERATE_METHODS[0]:
        USERINPUT_ImageData_1 = col1.file_uploader("Upload Start Image", ['png', 'jpg', 'jpeg', 'bmp'])
        if USERINPUT_ImageData_1 is not None:
            USERINPUT_ImageData_1 = USERINPUT_ImageData_1.read()
        else:
            USERINPUT_ImageData_1 = open(DEFAULT_PATH_EXAMPLEIMAGE1, 'rb').read()
        USERINPUT_Image_1 = cv2.imdecode(np.frombuffer(USERINPUT_ImageData_1, np.uint8), cv2.IMREAD_COLOR)
        USERINPUT_Image_1 = cv2.cvtColor(USERINPUT_Image_1, cv2.COLOR_BGR2RGB)
        # BGColor Simplify
        USERINPUT_BGColor, USERINPUT_Image_1 = UI_BGColorSelect(USERINPUT_Image_1, st=col1)
    # Random Image
    elif USERINPUT_GenerateMethod == GENERATE_METHODS[1]:
        # Get Size
        ImageSize = UI_CustomResize()
        ImageSize = np.array(list(ImageSize) + [3])
        N_Pixels = ImageSize[0]*ImageSize[1]
        minColorCount = min(10, N_Pixels)

        # Get Generation Parameters
        USERINPUT_NColors = col1.slider("N Random Colors", 1, int(N_Pixels/minColorCount), 1)
        USERINPUT_FillPercent = col1.slider("Image Fill Ratio", 0.0, 1.0, 1.0, 0.25)
        USERINPUT_BGColor = Hex_to_RGB(col1.color_picker("Select Background Color", value="#000000"))
        ColorCount_Range = (0, int((N_Pixels/USERINPUT_NColors)*USERINPUT_FillPercent))

        # Regen and Reshuffle Buttons
        col1, col2 = st.beta_columns(2)
        USERINPUT_Regenerate = col1.button("Regenerate")
        USERINPUT_Reshuffle = col2.button("Reshuffle")
        
        USERINPUT_Image_1 = None
        if USERINPUT_Regenerate:
            USERINPUT_Image_1 = GenerateRandomImage(ImageSize, USERINPUT_BGColor, USERINPUT_NColors, ColorCount_Range)
            cv2.imwrite(DEFAULT_SAVEPATH, USERINPUT_Image_1)
        else:
            USERINPUT_Image_1 = cv2.imread(DEFAULT_SAVEPATH)

    USERINPUT_Image_2 = None
    if USERINPUT_Reshuffle:
        USERINPUT_Image_2 = ImageGenerator.GenerateShuffledImage(USERINPUT_Image_1)
    else:
        USERINPUT_Image_2 = ShuffleImage_Cached(USERINPUT_Image_1)

    if USERINPUT_SwapImages:
        USERINPUT_Image_1, USERINPUT_Image_2 = USERINPUT_Image_2, USERINPUT_Image_1
    col1, col2 = st.beta_columns(2)
    col1.image(USERINPUT_Image_1, caption="Start Image", use_column_width=True)
    col2.image(USERINPUT_Image_2, caption="End Image", use_column_width=True)

    return USERINPUT_Image_1, USERINPUT_Image_2, np.array(USERINPUT_BGColor)


def UI_TransistionFuncSelect(title=''):
    TransistionFuncName = st.selectbox(title, list(TRANSISTIONFUNCS.keys()))
    TransistionFunc = TRANSISTIONFUNCS[TransistionFuncName]
    return TransistionFunc

def UI_LocationMappingFuncSelect():
    MappingFuncName = st.selectbox('Location Based Pixel Mapping Function', list(MAPPINGFUNCS_LOCATION.keys()))
    MappingFunc = MAPPINGFUNCS_LOCATION[MappingFuncName]
    return MappingFunc

def UI_ColorLocationMappingFuncSelect():
    MappingFuncName = st.selectbox('Color+Location Based Pixel Mapping Function', list(MAPPINGFUNCS_COLORLOCATION.keys()))
    MappingFunc = MAPPINGFUNCS_COLORLOCATION[MappingFuncName]
    return MappingFunc

def UI_CustomResize():
    col1, col2 = st.beta_columns(2)
    USERINPUT_ImageSizeY = col2.slider("Width Pixels", IMAGESIZE_MIN[0], IMAGESIZE_MAX[0], IMAGESIZE_DEFAULT[0], IMAGESIZE_MIN[0], key="USERINPUT_ImageSizeX")
    USERINPUT_ImageSizeX = col2.slider("Height Pixels", IMAGESIZE_MIN[1], IMAGESIZE_MAX[1], IMAGESIZE_DEFAULT[1], IMAGESIZE_MIN[1], key="USERINPUT_ImageSizeY")
    CustomSize = [int(USERINPUT_ImageSizeX), int(USERINPUT_ImageSizeY)]

    ImageSizeIndicator_Image = GenerateImageSizeIndicatorImage(CustomSize)
    col1.image(ImageSizeIndicator_Image, caption="Image Size (Max " + str(IMAGESIZE_MAX[0]) + " x " + str(IMAGESIZE_MAX[1]) + ")", use_column_width=False, clamp=False)
    
    return CustomSize

def UI_Resizer(USERINPUT_Image_1, USERINPUT_Image_2):
    ResizeFuncName = st.selectbox("Select Resize Method", list(RESIZEFUNCS.keys()))
    ResizeFunc = RESIZEFUNCS[ResizeFuncName]
    # Check for Custom Size
    if 'Custom Size' in ResizeFuncName:
        CustomSize = UI_CustomResize()
        ResizeFunc = functools.partial(ResizeFunc, Size=tuple(CustomSize))

    USERINPUT_Image_1, USERINPUT_Image_2 = ResizeFunc(USERINPUT_Image_1, USERINPUT_Image_2)
    col1, col2 = st.beta_columns(2)
    col1.image(USERINPUT_Image_1, caption="Resized Start Image", use_column_width=True)
    col2.image(USERINPUT_Image_2, caption="Resized End Image", use_column_width=True)

    return USERINPUT_Image_1, USERINPUT_Image_2

def UI_BGColorSelect(USERINPUT_Image_1, st=st):
    USERINPUT_BGColorStart = Hex_to_RGB(st.color_picker("Select Background Color Start", value="#000000"))
    USERINPUT_BGColorEnd = Hex_to_RGB(st.color_picker("Select Background Color End", value="#303030"))

    ## Simplify Images
    USERINPUT_Image_1 = ImageSimplify.ImageSimplify_RangeReplace(USERINPUT_Image_1, valRange=[USERINPUT_BGColorStart, USERINPUT_BGColorEnd], replaceVal=USERINPUT_BGColorStart)
    st.image(USERINPUT_Image_1, caption="Simplified Image", use_column_width=True)

    return np.array(USERINPUT_BGColorStart), USERINPUT_Image_1

# Repo Based Functions
def color_based_transistion():
    # Title
    st.header("Color Based Transistion")

    # Load Inputs
    USERINPUT_Image_1, USERINPUT_Image_2 = UI_LoadImageFiles()
    USERINPUT_Image_1, USERINPUT_Image_2 = UI_Resizer(USERINPUT_Image_1, USERINPUT_Image_2)
    TransistionFunc = UI_TransistionFuncSelect('Color Transistion Function')
    N = st.slider("Transistion Images Count", 3, 50, 10)
    USERINPUT_MirrorAnimation = st.checkbox("Mirror Animation")

    # Process Inputs
    if st.button("Generate"):
        GeneratedImgs = ColorBasedTransistion(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFunc, N)
        if USERINPUT_MirrorAnimation:
            GeneratedImgs = GeneratedImgs + GeneratedImgs[::-1]

        # Display Outputs
        UI_DisplayImageSequence(GeneratedImgs)

def location_based_transistion():
    # Title
    st.header("Location Based Transistion")

    # Load Inputs
    USERINPUT_Image_1, USERINPUT_Image_2, BGColor = UI_LoadGenerateImages()
    TransistionFunc_Y = UI_TransistionFuncSelect('X Transistion Function') # Flipped X and Y due to flipped coords in image indices
    TransistionFunc_X = UI_TransistionFuncSelect('Y Transistion Function') # Flipped X and Y due to flipped coords in image indices
    MappingFunc = UI_LocationMappingFuncSelect()
    
    N = st.slider("Transistion Images Count", 3, 50, 10)
    USERINPUT_MirrorAnimation = st.checkbox("Mirror Animation")

    # Process Inputs
    if st.button("Generate"):
        GeneratedImgs = LocationBasedTransistion(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFunc_X, TransistionFunc_Y, MappingFunc, N, BGColor)
        if USERINPUT_MirrorAnimation:
            GeneratedImgs = GeneratedImgs + GeneratedImgs[::-1]

        # Display Outputs
        UI_DisplayImageSequence(GeneratedImgs)
    
#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()
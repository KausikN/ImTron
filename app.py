"""
Stream lit GUI for hosting ImTron
"""

# Imports
import cv2
import functools
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import Img2ImgTransistion_ColorBased
import Img2ImgTransistion_LocationBased
import Img2ImgTransistion_LocationColorBased
import OtherTransistions

from Utils import MappingLibrary
from Utils import TransistionLibrary
from Utils import ResizeLibrary
from Utils import ImageSimplify
from Utils import ImageGenerator
from Utils import Utils

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
    DEFAULT_VIDEO_DURATION = st.sidebar.slider("Generated Video Duration", 0.1, 5.0, 2.0, 0.1)
    
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

DEFAULT_SAVEPATH1 = 'TestImgs/RandomImage1.png'
DEFAULT_SAVEPATH2 = 'TestImgs/RandomImage2.png'
DEFAULT_SAVEPATH_GIF = 'TestImgs/OutputGIF.gif'
DEFAULT_SAVEPATH_VIDEO = "StreamLitGUI/DefaultData/SavedVideo.avi"
DEFAULT_SAVEPATH_VIDEO_CONVERTED = "StreamLitGUI/DefaultData/SavedVideo_Converted.mp4"
DEFAULT_VIDEO_DURATION = 2.0

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

DISPLAY_IMAGESIZE = [512, 512]
DISPLAY_INTERPOLATION = cv2.INTER_NEAREST
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

@st.cache
def GeneratePixelPositionIndicatorImage(ImageSize, StartPosRel):
    ### Pixel Position Indicator Image
    displaySizeMax = max(ImageSize)
    displaySize = [int((ImageSize[0]/displaySizeMax)*IMAGESIZEINDICATORIMAGE_SIZE[0]), int((ImageSize[1]/displaySizeMax)*IMAGESIZEINDICATORIMAGE_SIZE[1])]
    StartPos = [int(StartPosRel[0]*displaySize[0]), int(StartPosRel[1]*displaySize[1])]
    PixelPosIndicator_Image = np.zeros(tuple(displaySize), dtype=int)
    padding = int((min(displaySize)/20)/2)
    horOffset = [max(StartPos[0]-padding, 0), min(StartPos[0]+padding, displaySize[0])]
    verOffset = [max(StartPos[1]-padding, 0), min(StartPos[1]+padding, displaySize[1])]
    PixelPosIndicator_Image[horOffset[0]:horOffset[1], verOffset[0]:verOffset[1]] = 255
    return PixelPosIndicator_Image

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

@st.cache
def LocationColorBasedTransistion(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFunc_X, TransistionFunc_Y, TransistionFunc_R, TransistionFunc_G, TransistionFunc_B, MappingFunc, N, BGColors):
    TransistionFuncs_Location = {
        'X': TransistionFunc_X,
        'Y': TransistionFunc_Y
    }
    TransistionFuncs_Color = {
        'R': TransistionFunc_R,
        'G': TransistionFunc_G,
        'B': TransistionFunc_B
    }
    BGColors = [[BGColors[0]], [BGColors[1]]]
    GeneratedImgs = Img2ImgTransistion_LocationColorBased.I2I_Transistion_LocationColorBased_Fast(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFuncs_Location, TransistionFuncs_Color, MappingFunc, N, BGColors)
    return GeneratedImgs

@st.cache
def SinglePixelTransistion(USERINPUT_Image_1, USERINPUT_StartPos, TransistionFunc_X, TransistionFunc_Y, N, BGColor):
    TransistionFuncs = {
        'X': TransistionFunc_X,
        'Y': TransistionFunc_Y
    }
    GeneratedImgs, StartImg = OtherTransistions.I_Transistion_SinglePixelExplode_Fast(USERINPUT_Image_1, USERINPUT_StartPos, TransistionFuncs, N, BGColor)
    return GeneratedImgs

# UI Functions
def UI_DisplayImageSequence(GeneratedImgs):
    # Resize
    use_column_width = True
    if DISPLAY_IMAGESIZE is not None:
        use_column_width = False
        displaySizeMax = max(GeneratedImgs[0].shape[0], GeneratedImgs[0].shape[1])
        displaySize = [int((DISPLAY_IMAGESIZE[1]/displaySizeMax)*GeneratedImgs[0].shape[1]), int((DISPLAY_IMAGESIZE[0]/displaySizeMax)*GeneratedImgs[0].shape[0])]
        print("Resizing Sequence for displaying...")
        for i in tqdm(range(len(GeneratedImgs))):
            GeneratedImgs[i] = cv2.resize(GeneratedImgs[i], tuple(displaySize), interpolation=DISPLAY_INTERPOLATION)

    outputDisplay = st.empty()
    while(True):
        for i in range(len(GeneratedImgs)):
            plt.pause(DISPLAY_DELAY)
            outputDisplay.image(GeneratedImgs[i], caption='Generated Transistion', use_column_width=use_column_width)

def UI_DisplayImageSequence_AsGIF(GeneratedImgs):
    GeneratedImgs_Display = []
    # Resize
    if DISPLAY_IMAGESIZE is not None:
        displaySizeMax = max(GeneratedImgs[0].shape[0], GeneratedImgs[0].shape[1])
        displaySize = [int((DISPLAY_IMAGESIZE[1]/displaySizeMax)*GeneratedImgs[0].shape[1]), int((DISPLAY_IMAGESIZE[0]/displaySizeMax)*GeneratedImgs[0].shape[0])]
        print("Resizing Sequence for displaying...")
        for I in tqdm(GeneratedImgs):
            GeneratedImgs_Display.append(cv2.resize(np.array(I), tuple(displaySize), interpolation=DISPLAY_INTERPOLATION))
    # Save and Display - Clear GIF
    Utils.SaveImageSequence(GeneratedImgs_Display, DEFAULT_SAVEPATH_GIF, mode='gif', frameSize=None, fps=25)
    st.image(DEFAULT_SAVEPATH_GIF, "Generated Transistion - Clear Display", use_column_width=False)
    # Save and Display - Output Video
    Utils.SaveImageSequence(GeneratedImgs, DEFAULT_SAVEPATH_GIF, mode='gif', frameSize=None, fps=25)
    st.image(DEFAULT_SAVEPATH_GIF, "Generated Transistion - Actual Size", use_column_width=False)

def UI_DisplayImageSequence_AsVideo(GeneratedImgs):
    GeneratedImgs_Display = []
    # Resize
    if DISPLAY_IMAGESIZE is not None:
        displaySizeMax = max(GeneratedImgs[0].shape[0], GeneratedImgs[0].shape[1])
        displaySize = [int((DISPLAY_IMAGESIZE[1]/displaySizeMax)*GeneratedImgs[0].shape[1]), int((DISPLAY_IMAGESIZE[0]/displaySizeMax)*GeneratedImgs[0].shape[0])]
        print("Resizing Sequence for displaying...")
        for I in tqdm(GeneratedImgs):
            GeneratedImgs_Display.append(cv2.resize(np.array(I), tuple(displaySize), interpolation=DISPLAY_INTERPOLATION))
    # Save and Display - Clear GIF
    Utils.SaveImageSequence(GeneratedImgs_Display, DEFAULT_SAVEPATH_GIF, mode='gif', frameSize=None, fps=25)
    st.image(DEFAULT_SAVEPATH_GIF, "Generated Transistion - Clear Display", use_column_width=False)
    # Save and Display - Output Video
    st.markdown("## Generated Transistion Video")
    fps = (len(GeneratedImgs_Display)/DEFAULT_VIDEO_DURATION)
    Utils.SaveFrames2Video(GeneratedImgs_Display, DEFAULT_SAVEPATH_VIDEO, fps)
    Utils.FixVideoFile(DEFAULT_SAVEPATH_VIDEO, DEFAULT_SAVEPATH_VIDEO_CONVERTED)
    st.video(DEFAULT_SAVEPATH_VIDEO_CONVERTED)

def UI_LoadImageFiles():
    USERINPUT_SwapImages = st.checkbox("Swap Images")
    col1, col2 = st.columns(2)

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

def UI_LoadGenerateImages_Location():
    global REGEN_KEY

    st.markdown("### Generate Images")

    USERINPUT_Reshuffle = False
    USERINPUT_Regenerate = False

    USERINPUT_SwapImages = st.checkbox("Swap Images")
    USERINPUT_BGColor = []

    GENERATE_METHODS = ["Upload Image", "Generate Random Image"]
    USERINPUT_GenerateMethod = st.selectbox("Select Generate Method", GENERATE_METHODS)
    # Upload
    if USERINPUT_GenerateMethod == GENERATE_METHODS[0]:
        USERINPUT_ImageData_1 = st.file_uploader("Upload Start Image", ['png', 'jpg', 'jpeg', 'bmp'])
        if USERINPUT_ImageData_1 is not None:
            USERINPUT_ImageData_1 = USERINPUT_ImageData_1.read()
        else:
            USERINPUT_ImageData_1 = open(DEFAULT_PATH_EXAMPLEIMAGE1, 'rb').read()
        USERINPUT_Image_1 = cv2.imdecode(np.frombuffer(USERINPUT_ImageData_1, np.uint8), cv2.IMREAD_COLOR)
        USERINPUT_Image_1 = cv2.cvtColor(USERINPUT_Image_1, cv2.COLOR_BGR2RGB)
        # Resize and Simplify
        CustomSize = UI_CustomResize()
        USERINPUT_Image_1 = cv2.resize(USERINPUT_Image_1, tuple(CustomSize))
        col1, col2 = st.columns(2)
        USERINPUT_BGColorStart = Hex_to_RGB(col1.color_picker("Select Background Color Start", value="#000000", key="Start" + str(col1)))
        USERINPUT_BGColorEnd = Hex_to_RGB(col2.color_picker("Select Background Color End", value="#303030", key="End" + str(col2)))
        col1.image(USERINPUT_Image_1, caption="Resized Source Image", use_column_width=True)
        USERINPUT_Image_1 = ImageSimplify.ImageSimplify_RangeReplace(USERINPUT_Image_1, valRange=[USERINPUT_BGColorStart, USERINPUT_BGColorEnd], replaceVal=USERINPUT_BGColorStart)
        col2.image(USERINPUT_Image_1, caption="Simplified Image", use_column_width=True)
        USERINPUT_BGColor = USERINPUT_BGColorStart

        # Regen and Reshuffle Buttons
        col1, col2 = st.columns(2)
        USERINPUT_Reshuffle = col2.button("Reshuffle")

    # Random Image
    elif USERINPUT_GenerateMethod == GENERATE_METHODS[1]:
        # Get Size
        ImageSize = UI_CustomResize()
        ImageSize = np.array(list(ImageSize) + [3])
        N_Pixels = ImageSize[0]*ImageSize[1]
        minColorCount = min(10, N_Pixels)

        # Get Generation Parameters
        USERINPUT_NColors = st.slider("N Random Colors", 1, int(N_Pixels/minColorCount), 1)
        USERINPUT_FillPercents = st.slider("Image Fill Ratio", 0.0, 1.0, (0.0, 1.0), 0.1)
        USERINPUT_BGColor = Hex_to_RGB(st.color_picker("Select Background Color", value="#000000"))
        ColorCount_Range = (int((N_Pixels/USERINPUT_NColors)*USERINPUT_FillPercents[0]), int((N_Pixels/USERINPUT_NColors)*USERINPUT_FillPercents[1]))

        # Regen and Reshuffle Buttons
        col1, col2 = st.columns(2)
        USERINPUT_Regenerate = col1.button("Regenerate")
        USERINPUT_Reshuffle = col2.button("Reshuffle")
        
        USERINPUT_Image_1 = None
        if USERINPUT_Regenerate:
            USERINPUT_Image_1 = GenerateRandomImage(ImageSize, USERINPUT_BGColor, USERINPUT_NColors, ColorCount_Range)
            cv2.imwrite(DEFAULT_SAVEPATH1, USERINPUT_Image_1)
        else:
            USERINPUT_Image_1 = cv2.imread(DEFAULT_SAVEPATH1)

    USERINPUT_Image_2 = None
    if USERINPUT_Reshuffle:
        USERINPUT_Image_2 = ImageGenerator.GenerateShuffledImage(USERINPUT_Image_1)
    else:
        USERINPUT_Image_2 = ShuffleImage_Cached(USERINPUT_Image_1)

    if USERINPUT_SwapImages:
        USERINPUT_Image_1, USERINPUT_Image_2 = USERINPUT_Image_2, USERINPUT_Image_1
    col1, col2 = st.columns(2)
    col1.image(USERINPUT_Image_1, caption="Start Image", use_column_width=True)
    col2.image(USERINPUT_Image_2, caption="End Image", use_column_width=True)

    return USERINPUT_Image_1, USERINPUT_Image_2, np.array(USERINPUT_BGColor)

def UI_LoadGenerateImages_LocationColor():
    global REGEN_KEY

    st.markdown("### Generate Images")

    USERINPUT_Regenerate = False

    USERINPUT_SwapImages = st.checkbox("Swap Images")
    USERINPUT_BGColor = []

    GENERATE_METHODS = ["Upload Image", "Generate Random Image"]
    USERINPUT_GenerateMethod = st.selectbox("Select Generate Method", GENERATE_METHODS)

    USERINPUT_Image_1 = None
    USERINPUT_Image_2 = None
    # Upload
    if USERINPUT_GenerateMethod == GENERATE_METHODS[0]:
        col1, col2 = st.columns(2)
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
        # Resize
        USERINPUT_Image_1, USERINPUT_Image_2 = UI_Resizer(USERINPUT_Image_1, USERINPUT_Image_2)
        # BGColor Simplify
        col1, col2 = st.columns(2)
        USERINPUT_BGColor_1, USERINPUT_Image_1 = UI_BGColorSelect(USERINPUT_Image_1, col=col1)
        USERINPUT_BGColor_2, USERINPUT_Image_2 = UI_BGColorSelect(USERINPUT_Image_2, col=col2)
        USERINPUT_BGColor = [USERINPUT_BGColor_1, USERINPUT_BGColor_2]
        
    # Random Image
    elif USERINPUT_GenerateMethod == GENERATE_METHODS[1]:
        # Get Size
        ImageSize = UI_CustomResize()
        ImageSize = np.array(list(ImageSize) + [3])
        N_Pixels = ImageSize[0]*ImageSize[1]
        minColorCount = min(10, N_Pixels)

        # Get Generation Parameters 1
        col1, col2 = st.columns(2)
        USERINPUT_NColors_1 = col1.slider("N Random Colors", 1, int(N_Pixels/minColorCount), 1)
        USERINPUT_FillPercents_1 = col1.slider("Image Fill Ratio", 0.0, 1.0, (0.0, 1.0), 0.1)
        USERINPUT_BGColor_1 = Hex_to_RGB(col1.color_picker("Select Background Color", value="#000000"))
        ColorCount_Range_1 = (int((N_Pixels/USERINPUT_NColors_1)*USERINPUT_FillPercents_1[0]), int((N_Pixels/USERINPUT_NColors_1)*USERINPUT_FillPercents_1[1]))
        USERINPUT_Regenerate_1 = col1.button("Regenerate")
        if USERINPUT_Regenerate_1:
            USERINPUT_Image_1 = GenerateRandomImage(ImageSize, USERINPUT_BGColor_1, USERINPUT_NColors_1, ColorCount_Range_1)
            cv2.imwrite(DEFAULT_SAVEPATH1, USERINPUT_Image_1)
        else:
            USERINPUT_Image_1 = cv2.imread(DEFAULT_SAVEPATH1)
        # Get Generation Parameters 2
        USERINPUT_NColors_2 = col2.slider("N Random Colors", 1, int(N_Pixels/minColorCount), 1, key='2')
        USERINPUT_FillPercents_2 = col2.slider("Image Fill Ratio", 0.0, 1.0, (0.0, 1.0), 0.1, key='2')
        USERINPUT_BGColor_2 = Hex_to_RGB(col2.color_picker("Select Background Color", value="#000000", key='2'))
        ColorCount_Range_2 = (int((N_Pixels/USERINPUT_NColors_2)*USERINPUT_FillPercents_2[0]), int((N_Pixels/USERINPUT_NColors_2)*USERINPUT_FillPercents_2[1]))
        USERINPUT_Regenerate_2 = col2.button("Regenerate", key='2')
        if USERINPUT_Regenerate_2:
            USERINPUT_Image_2 = GenerateRandomImage(ImageSize, USERINPUT_BGColor_2, USERINPUT_NColors_2, ColorCount_Range_2)
            cv2.imwrite(DEFAULT_SAVEPATH2, USERINPUT_Image_2)
        else:
            USERINPUT_Image_2 = cv2.imread(DEFAULT_SAVEPATH2)
        USERINPUT_BGColor = [USERINPUT_BGColor_1, USERINPUT_BGColor_2]

    if USERINPUT_SwapImages:
        USERINPUT_Image_1, USERINPUT_Image_2 = USERINPUT_Image_2, USERINPUT_Image_1
    
    col1, col2 = st.columns(2)
    col1.image(USERINPUT_Image_1, caption="Start Image", use_column_width=True)
    col2.image(USERINPUT_Image_2, caption="End Image", use_column_width=True)

    return USERINPUT_Image_1, USERINPUT_Image_2, np.array(USERINPUT_BGColor)

def UI_LoadGenerateImages_SinglePixel():
    global REGEN_KEY

    st.markdown("### Generate Image")

    USERINPUT_Regenerate = False
    USERINPUT_BGColor = []

    GENERATE_METHODS = ["Upload Image", "Generate Random Image"]
    USERINPUT_GenerateMethod = st.selectbox("Select Generate Method", GENERATE_METHODS)
    USERINPUT_Image_1 = None
    USERINPUT_Image_2 = None
    # Upload
    if USERINPUT_GenerateMethod == GENERATE_METHODS[0]:
        USERINPUT_ImageData_1 = st.file_uploader("Upload Start Image", ['png', 'jpg', 'jpeg', 'bmp'])
        if USERINPUT_ImageData_1 is not None:
            USERINPUT_ImageData_1 = USERINPUT_ImageData_1.read()
        else:
            USERINPUT_ImageData_1 = open(DEFAULT_PATH_EXAMPLEIMAGE1, 'rb').read()
        USERINPUT_Image_1 = cv2.imdecode(np.frombuffer(USERINPUT_ImageData_1, np.uint8), cv2.IMREAD_COLOR)
        USERINPUT_Image_1 = cv2.cvtColor(USERINPUT_Image_1, cv2.COLOR_BGR2RGB)
        # Resize and Simplify
        CustomSize = UI_CustomResize()
        USERINPUT_Image_1 = cv2.resize(USERINPUT_Image_1, tuple(CustomSize))
        col1, col2 = st.columns(2)
        USERINPUT_BGColorStart = Hex_to_RGB(col1.color_picker("Select Background Color Start", value="#000000", key="Start" + str(col1)))
        USERINPUT_BGColorEnd = Hex_to_RGB(col2.color_picker("Select Background Color End", value="#303030", key="End" + str(col2)))
        col1.image(USERINPUT_Image_1, caption="Resized Source Image", use_column_width=True)
        USERINPUT_Image_1 = ImageSimplify.ImageSimplify_RangeReplace(USERINPUT_Image_1, valRange=[USERINPUT_BGColorStart, USERINPUT_BGColorEnd], replaceVal=USERINPUT_BGColorStart)
        col2.image(USERINPUT_Image_1, caption="Simplified Image", use_column_width=True)
        USERINPUT_BGColor = list(USERINPUT_BGColorStart)
    # Random Image
    elif USERINPUT_GenerateMethod == GENERATE_METHODS[1]:
        # Get Size
        ImageSize = UI_CustomResize()
        ImageSize = np.array(list(ImageSize) + [3])
        N_Pixels = ImageSize[0]*ImageSize[1]
        minColorCount = min(10, N_Pixels)

        # Get Generation Parameters
        USERINPUT_NColors = st.slider("N Random Colors", 1, int(N_Pixels/minColorCount), 1)
        USERINPUT_FillPercents = st.slider("Image Fill Ratio", 0.0, 1.0, (0.0, 1.0), 0.1)
        USERINPUT_BGColor = Hex_to_RGB(st.color_picker("Select Background Color", value="#000000"))
        ColorCount_Range = (int((N_Pixels/USERINPUT_NColors)*USERINPUT_FillPercents[0]), int((N_Pixels/USERINPUT_NColors)*USERINPUT_FillPercents[1]))

        # Regen Button
        USERINPUT_Regenerate = st.button("Regenerate")

        USERINPUT_Image_1 = None
        if USERINPUT_Regenerate:
            USERINPUT_Image_1 = GenerateRandomImage(ImageSize, USERINPUT_BGColor, USERINPUT_NColors, ColorCount_Range)
            cv2.imwrite(DEFAULT_SAVEPATH1, USERINPUT_Image_1)
        else:
            USERINPUT_Image_1 = cv2.imread(DEFAULT_SAVEPATH1)
        
        col1, col2 = st.columns(2)
        col1.image(USERINPUT_Image_1, caption="Generated Source Image", use_column_width=True)

    USERINPUT_StartPos = UI_SelectStartPosition(USERINPUT_Image_1.shape[:2])

    return USERINPUT_Image_1, USERINPUT_StartPos, np.array(USERINPUT_BGColor)

def UI_TransistionFuncSelect(title='', col=st):
    TransistionFuncName = col.selectbox(title, list(TRANSISTIONFUNCS.keys()))
    TransistionFunc = TRANSISTIONFUNCS[TransistionFuncName]
    return TransistionFunc

def UI_LocationMappingFuncSelect():
    MappingFuncName = st.selectbox('Location Based Pixel Mapping Function', list(MAPPINGFUNCS_LOCATION.keys()))
    MappingFunc = MAPPINGFUNCS_LOCATION[MappingFuncName]
    return MappingFunc

def UI_ColorLocationMappingFuncSelect():
    MappingFuncName = st.selectbox('Color+Location Based Pixel Mapping Function', list(MAPPINGFUNCS_COLORLOCATION.keys()))
    MappingFunc = MAPPINGFUNCS_COLORLOCATION[MappingFuncName]
    if MappingFuncName == 'Distance Based':
        options={"C_L_Ratio": 0.5, "ColorSign": 1, "LocationSign": 1}
        options["C_L_Ratio"] = st.slider("Color to Location Ratio", 0.0, 1.0, 0.5, 0.1)
        col1, col2 = st.columns(2)
        options["ColorSign"] = -1.0 if ((col1.select_slider("Color Optimizer", ["Minimize", "Maximize"], "Minimize")) == "Maximize") else 1.0
        options["LocationSign"] = -1.0 if ((col2.select_slider("Location Optimizer", ["Minimize", "Maximize"], "Minimize")) == "Maximize") else 1.0
        MappingFunc = functools.partial(MappingFunc, options=options)
    return MappingFunc

def UI_SelectStartPosition(ImageSize):
    col1, col2 = st.columns(2)
    USERINPUT_StartPosRelY = col2.slider("Horizontal", 0.0, 1.0, 0.5, 0.1, key="USERINPUT_StartPosRelX")
    USERINPUT_StartPosRelX = col2.slider("Vertical", 0.0, 1.0, 0.5, 0.1, key="USERINPUT_StartPosRelY")
    StartPosRel = [USERINPUT_StartPosRelX, USERINPUT_StartPosRelY]
    StartPos = [int(ImageSize[0]*USERINPUT_StartPosRelX), int(ImageSize[1]*USERINPUT_StartPosRelY)]
    PixelPosIndicator_Image = GeneratePixelPositionIndicatorImage(ImageSize, StartPosRel)
    col1.image(PixelPosIndicator_Image, caption="Start Pixel Position", use_column_width=True, clamp=False)
    
    return StartPos

def UI_CustomResize():
    col1, col2 = st.columns(2)
    USERINPUT_ImageSizeX = col2.slider("Width Pixels", IMAGESIZE_MIN[0], IMAGESIZE_MAX[0], IMAGESIZE_DEFAULT[0], IMAGESIZE_MIN[0], key="USERINPUT_ImageSizeX")
    USERINPUT_ImageSizeY = col2.slider("Height Pixels", IMAGESIZE_MIN[1], IMAGESIZE_MAX[1], IMAGESIZE_DEFAULT[1], IMAGESIZE_MIN[1], key="USERINPUT_ImageSizeY")
    CustomSize = [int(USERINPUT_ImageSizeX), int(USERINPUT_ImageSizeY)]

    ImageSizeIndicator_Image = GenerateImageSizeIndicatorImage(CustomSize[::-1]) # Reversed due to dissimilarity in generating width and height
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
    col1, col2 = st.columns(2)
    col1.image(USERINPUT_Image_1, caption="Resized Start Image", use_column_width=True)
    col2.image(USERINPUT_Image_2, caption="Resized End Image", use_column_width=True)

    return USERINPUT_Image_1, USERINPUT_Image_2

def UI_BGColorSelect(USERINPUT_Image_1, col=st):
    USERINPUT_BGColorStart = Hex_to_RGB(col.color_picker("Select Background Color Start", value="#000000", key="Start" + str(col)))
    USERINPUT_BGColorEnd = Hex_to_RGB(col.color_picker("Select Background Color End", value="#303030", key="End" + str(col)))

    ## Simplify Images
    USERINPUT_Image_1 = ImageSimplify.ImageSimplify_RangeReplace(USERINPUT_Image_1, valRange=[USERINPUT_BGColorStart, USERINPUT_BGColorEnd], replaceVal=USERINPUT_BGColorStart)
    col.image(USERINPUT_Image_1, caption="Simplified Image", use_column_width=True)

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
        UI_DisplayImageSequence_AsGIF(GeneratedImgs)

def location_based_transistion():
    # Title
    st.header("Location Based Transistion")

    # Load Inputs
    USERINPUT_Image_1, USERINPUT_Image_2, BGColor = UI_LoadGenerateImages_Location()
    col1, col2 = st.columns(2)
    TransistionFunc_Y = UI_TransistionFuncSelect('X Transistion Function', col1) # Flipped X and Y due to flipped coords in image indices
    TransistionFunc_X = UI_TransistionFuncSelect('Y Transistion Function', col2) # Flipped X and Y due to flipped coords in image indices
    MappingFunc = UI_LocationMappingFuncSelect()
    
    N = st.slider("Transistion Images Count", 3, 50, 10)
    USERINPUT_ReverseAnimation = st.checkbox("Reverse Animation")

    # Process Inputs
    if st.button("Generate"):
        GeneratedImgs = LocationBasedTransistion(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFunc_X, TransistionFunc_Y, MappingFunc, N, BGColor)
        if USERINPUT_ReverseAnimation:
            GeneratedImgs = GeneratedImgs[::-1]

        # Display Outputs
        UI_DisplayImageSequence_AsGIF(GeneratedImgs)

def image_to_image_transistion():
    # Title
    st.header("Image to Image Transistion")

    # Load Inputs
    USERINPUT_Image_1, USERINPUT_Image_2, BGColors = UI_LoadGenerateImages_LocationColor()
    col1, col2 = st.columns(2)
    TransistionFunc_Y = UI_TransistionFuncSelect('X Transistion Function', col1) # Flipped X and Y due to flipped coords in image indices
    TransistionFunc_X = UI_TransistionFuncSelect('Y Transistion Function', col2) # Flipped X and Y due to flipped coords in image indices
    col1, col2, col3 = st.columns(3)
    TransistionFunc_R = UI_TransistionFuncSelect('Red Transistion Function', col1)
    TransistionFunc_G = UI_TransistionFuncSelect('Green Transistion Function', col2)
    TransistionFunc_B = UI_TransistionFuncSelect('Blue Transistion Function', col3)
    MappingFunc = UI_ColorLocationMappingFuncSelect()
    
    N = st.slider("Transistion Images Count", 3, 50, 10)
    USERINPUT_ReverseAnimation = st.checkbox("Reverse Animation")

    # Process Inputs
    if st.button("Generate"):
        GeneratedImgs = LocationColorBasedTransistion(USERINPUT_Image_1, USERINPUT_Image_2, TransistionFunc_X, TransistionFunc_Y, TransistionFunc_R, TransistionFunc_G, TransistionFunc_B, MappingFunc, N, BGColors)
        if USERINPUT_ReverseAnimation:
            GeneratedImgs = GeneratedImgs[::-1]

        # Display Outputs
        UI_DisplayImageSequence_AsGIF(GeneratedImgs)

def single_pixel_transistion():
    # Title
    st.header("Single Pixel Transistion")

    # Load Inputs
    USERINPUT_Image_1, USERINPUT_StartPos, BGColor = UI_LoadGenerateImages_SinglePixel()
    col1, col2 = st.columns(2)
    TransistionFunc_Y = UI_TransistionFuncSelect('X Transistion Function', col1) # Flipped X and Y due to flipped coords in image indices
    TransistionFunc_X = UI_TransistionFuncSelect('Y Transistion Function', col2) # Flipped X and Y due to flipped coords in image indices
    
    N = st.slider("Transistion Images Count", 3, 50, 10)
    USERINPUT_ReverseAnimation = st.checkbox("Reverse Animation")

    # Process Inputs
    if st.button("Generate"):
        GeneratedImgs = SinglePixelTransistion(USERINPUT_Image_1, USERINPUT_StartPos, TransistionFunc_X, TransistionFunc_Y, N, BGColor)
        if USERINPUT_ReverseAnimation:
            GeneratedImgs = GeneratedImgs[::-1]

        # Display Outputs
        UI_DisplayImageSequence_AsGIF(GeneratedImgs)
    
#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()
'''
Library for Gradient Colour Generation and Viewing
'''

# Imports
import cv2
from colour import Color
import numpy as np
from numpy import random as rnd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

# Main Functions

################# - IMPLEMENTATION 1 - #####################################################################
# Explanation at https://bsou.io/posts/color-gradients-with-python
def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
        "r":[RGB[0] for RGB in gradient],
        "g":[RGB[1] for RGB in gradient],
        "b":[RGB[2] for RGB in gradient]}

# Uni-Linear Gradient - Straight line gradient
def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' Returns a gradient list of (n) colors between two hex colors. 
    start_hex and finish_hex should be the full six-digit color string,
    including the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
        int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
        for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)

# Poly-Linear Gradient - Multi Line Gradient
def rand_hex_color(num=1):
    ''' Generate random hex colors, default is one,
        returning a string. If num is greater than
        1, an array of strings is returned. '''
    colors = [
        RGB_to_hex([x*255 for x in rnd.rand(3)])
        for i in range(num)
    ]
    if num == 1:
        return colors[0]
    else:
        return colors

def polylinear_gradient(colors, n):
    ''' returns a list of colors forming linear gradients between
        all sequential pairs of colors. "n" specifies the total
        number of desired output colors '''
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col+1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]

    return gradient_dict

# Bezier Gradient
# Value cache
fact_cache = {}
def fact(n):
    ''' Memoized factorial function '''
    try:
        return fact_cache[n]
    except(KeyError):
        if n in [0, 1]:
            result = 1
        else:
            result = n*fact(n-1)
        fact_cache[n] = result
        return result


def bernstein(t, n, i):
    ''' Bernstein coefficient '''
    binom = fact(n)/float(fact(i)*fact(n - i))
    return binom*((1-t)**(n-i))*(t**i)


def bezier_gradient(colors, n_out=100):
    ''' Returns a "bezier gradient" dictionary
        using a given list of colors as control
        points. Dictionary also contains control
        colors/points. '''
    # RGB vectors for each color, use as control points
    RGB_list = [hex_to_RGB(color) for color in colors]
    n = len(RGB_list) - 1

    def bezier_interp(t):
        ''' Define an interpolation function
            for this specific curve'''
        # List of all summands
        summands = [
            map(lambda x: int(bernstein(t,n,i)*x), c)
            for i, c in enumerate(RGB_list)
        ]
        # Output color
        out = [0,0,0]
        # Add components of each summand together
        for vector in summands:
            for c in range(3):
                out[c] += vector[c]

        return out

    gradient = [
        bezier_interp(float(t)/(n_out-1))
        for t in range(n_out)
    ]
    # Return all points requested for gradient
    return {
        "gradient": color_dict(gradient),
        "control": color_dict(RGB_list)
    }

# Plotting
def plot_gradient_series(color_dict, filename, pointsize=100, control_points=None):
    ''' Take a dictionary containing the color
        gradient in RBG and hex form and plot
        it to a 3D matplotlib device '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xcol = color_dict["r"]
    ycol = color_dict["g"]
    zcol = color_dict["b"]

    # We can pass a vector of colors
    # corresponding to each point
    ax.scatter(xcol, ycol, zcol, c=color_dict["hex"], s=pointsize)

    # If bezier control points passed to function,
    # plot along with curve
    if not control_points == None:
        xcntl = control_points["r"]
        ycntl = control_points["g"]
        zcntl = control_points["b"]
        ax.scatter( xcntl, ycntl, zcntl, c=control_points["hex"], s=pointsize, marker='s')

    ax.set_xlabel('Red Value')
    ax.set_ylabel('Green Value')
    ax.set_zlabel('Blue Value')
    ax.set_zlim3d(0, 255)
    plt.ylim(0, 255)
    plt.xlim(0, 255)

    # Save two views of each plot
    ax.view_init(elev=15, azim=68)
    plt.savefig(filename + ".svg")
    ax.view_init(elev=15, azim=28)
    plt.savefig(filename + "_view_2.svg")

    # Show plot for testing
    plt.show()
################# - IMPLEMENTATION 1 - #####################################################################

################# - IMPLEMENTATION 2 - #####################################################################
class Gradient:
    def __init__(self, color_list, angle):
        self.color_list = color_list
        self.angle = angle


def LinearGradient_Library(start_color, end_color, n):
    return list(start_color.range_to(end_color, n))

def Hex2RGB(color):
    return list(mpl.colors.to_rgb(color))

def RGB2Hex(color):
    return mpl.colors.to_hex(color)

def RotateImage(I, angle, filled=False, tqdmDisable=True):
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
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def LineGradient_Linear(start_color, end_color, n, tqdmDisable=True):
    gradient = []
    for x in tqdm(range(n+1), disable=tqdmDisable):
        gradient.append(ColorMix(start_color, end_color, x/n))
    return gradient

def LineGradient_Display(gradient, linewidth=4, figsize=(8, 5), tqdmDisable=True):
    fig, ax = plt.subplots(figsize=figsize)
    for gi in tqdm(range(len(gradient)), disable=tqdmDisable):
        ax.axvline(gi+1, color=gradient[gi], linewidth=linewidth) 
    plt.show()

def LineGradient2Image(gradient, imgsize, filled=True):
    # Convert to rgb
    for i in range(len(gradient.color_list)):
        gradient.color_list[i] = list(mpl.colors.to_rgb(gradient.color_list[i]))

    I = np.array([gradient.color_list]*len(gradient.color_list))
    I = cv2.resize(I, imgsize)

    I = RotateImage(I, gradient.angle, filled=filled)

    return I

def AddGradients(g1, g2, imgsize):
    I1 = LineGradient2Image(g1, imgsize)
    I2 = LineGradient2Image(g2, imgsize)

    return I1 + I2

# Driver Code

"""
# Simple Gradient
start_color = '#f00'
end_color = '#00f'
n = 5000


gradient = Gradient(None, 0)
gradient.color_list = LineGradient_Linear(start_color, end_color, n)

angles = [0, 30, 45, 60, 90]
nCols = 5

nRows = round(len(angles) / nCols)
for anglei in tqdm(range(len(angles))):
    gradient.angle = angles[anglei]
    I = LineGradient2Image(gradient, (100, 100))
    plt.subplot(nRows, nCols, anglei+1)
    plt.imshow(I)
plt.show()
# LineGradient_Display(gradient)
"""

"""
# Adding Gradients
imgsize = (100, 100)
n = 200
nCols = 5

gradients = []
gradients.append(Gradient(LineGradient_Linear('#f00', '#0f0', n), 0))
gradients.append(Gradient(LineGradient_Linear('#00f', '#0f0', n), 45))
gradients.append(Gradient(LineGradient_Linear('#f0f', '#000', n), 0))
gradients.append(Gradient(LineGradient_Linear('#033', '#139', n), 90))

Is = []
IC = np.zeros(1)
for gradient in tqdm(gradients, disable=False):
    Is.append(LineGradient2Image(gradient, imgsize))
    if IC.ndim == 1:
        IC = Is[-1]
    else:
        IC = IC + Is[-1]
IC = IC * 255 / len(Is)
IC = IC.astype(int)

nRows = round(len(Is) / nCols) + 1
for i in range(len(Is)):
    plt.subplot(nRows, nCols, i+1)
    plt.imshow(Is[i])
plt.subplot(nRows, nCols, nCols*(nRows-1) + 1)
plt.imshow(IC)
plt.show()
"""
'''
Image Simplification Functions
'''

# Imports
import numpy as np

# Main Functions
# Range Replace Method
def ImageSimplify_RangeReplace(I, valRange=[[0, 0, 0], [50, 50, 50]], replaceVal=[0, 0, 0]):
    '''
    Image Simplify using Range Replace Method
    '''
    I_simplified = np.copy(I)
    # Generate Replace Mask
    I_replaceMask = (I >= valRange[0]) * (I <= valRange[1])
    I_replaceMask = np.logical_and(*tuple([I_replaceMask[:, :, i] for i in range(I.shape[-1])]))
    # Replace Values
    I_simplified[I_replaceMask] = replaceVal
    return I_simplified

# Driver Code
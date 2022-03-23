'''
Transisition Functions
'''

# Imports
import numpy as np

# Main Functions
def Transistion_Linear(v1, v2, N, **params):
    '''
    Linear Transistion
    '''
    # Fix Data
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Transistion
    C_Gen = np.linspace(v1, v2, N).astype(int)

    return C_Gen

# Main Vars
TRANSISTION_FUNCS = {
    "Linear": Transistion_Linear
}
NORMALISER_FUNCS = {
    "Clip": "clip",
    "Average": "avg"
}

# Driver Code
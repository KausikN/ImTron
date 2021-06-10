'''
Library Containing many Transisition Functions
'''

# Imports
import numpy as np

# Main Functions
# Slow Transistion Functions
def LinearTransistion(c1, c2, N, options=None):
    C_Gen = None
    # If colours
    if type(c1) == type([]):
        C_Gen = np.zeros((N, len(c1)))
        for i in range(len(c1)):
            C_Gen[:, i] = np.linspace(c1[i], c2[i], N).astype(int)
    else:
        C_Gen = np.linspace(c1, c2, N).astype(int)
    return list(C_Gen)

# Numpy accelerated Transistion Functions
def LinearTransistion_Fast(c1, c2, N, options=None):
    c1 = np.array(c1)
    c2 = np.array(c2)

    C_Gen = np.linspace(c1, c2, N).astype(int)
    return C_Gen
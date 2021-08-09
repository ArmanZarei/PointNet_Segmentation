import numpy as np


def read_pts(file):
    return np.genfromtxt(file)

def read_seg(file):
    return np.genfromtxt(file, dtype=(int))
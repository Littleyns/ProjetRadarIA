import numpy as np
from scipy import signal

def convolve_using(kernel, sig):
    filtered = signal.convolve(sig, kernel, mode="same") #/ sum(kernel)
    return filtered
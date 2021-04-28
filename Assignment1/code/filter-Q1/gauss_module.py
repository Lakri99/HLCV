# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import scipy
from scipy.signal import convolve2d as conv2

def gauss(sigma):
    x = np.arange(-3*sigma,(3*sigma)+1)
    #x = x*sigma
    #print(x)
    first_term = (1/ (np.sqrt(2*np.pi)*sigma))
    second_term = np.exp(-(np.square(x)/(2*(sigma**2))))
    Gx = first_term * second_term
    #Gx = Gx/sum(Gx)
    return Gx, x

def gaussianfilter(img, sigma):
    [gx,x] = gauss(sigma)
    gx_2d = np.outer(gx,gx) # 2d Gaussian kernel
    normalize_factor = sum(sum(gx_2d))
    gx_2d = gx_2d/normalize_factor
    outimage = conv2(img, gx_2d, boundary='symm', mode='same')
    return outimage

def gaussdx(sigma):
    x = np.arange(-3*sigma,(3*sigma)+1)
    #x = x*sigma
    #print(x)
    first_term = -(1/ (np.sqrt(2*np.pi)*(sigma**3)))
    second_term = x * np.exp(-(np.square(x)/(2*(sigma**2))))
    D = first_term * second_term
    return D, x

def gaussderiv(img, sigma):
    [Gx,x] = gauss(sigma)
    [D,x] = gaussdx(sigma)
    G = Gx.reshape(1,Gx.size)
    D = D.reshape(1,D.size)
    imgDx = conv2(conv2(img, G.T, 'same'), D, 'same')
    imgDy = conv2(conv2(img, G,'same'), D.T, 'same')
    return imgDx, imgDy

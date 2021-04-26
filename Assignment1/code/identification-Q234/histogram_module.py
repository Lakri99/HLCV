import numpy as np
from numpy import histogram as hist

#  Compute the equal width bin interval
def bin_interval(num_bins):
    interval_size = 255/num_bins
    interval = []
    temp = 0
    for i in range(num_bins):
      interval.append([temp, temp + interval_size])
      temp += interval_size

    return interval

#  normalise an array
def normalize(x):
    normalized_test_array = (x - min(x)) / (max(x) - min(x)) 
    return normalized_test_array


#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    hists = np.zeros(num_bins)
    interval = bin_interval(num_bins)
    for i in img_gray.flatten():
      for pos, bins in enumerate(interval):
        if i >= bins[0] and i < bins[1]:
          hists[pos] += 1
    hists = normalize(hists)
    bins = [i[0] for i in interval]
    bins.append(float(255))
    # your code here
    return hists, np.asarray(bins, dtype=np.float32)

#  compute joint histogram for each color channel in the image, histogram should be normalized so that sum of all values equals 1
#  assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
def rgb_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    interval = bin_interval(num_bins)

    # execute the loop for each pixel in the image 
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            # ...
            R,G,B = img_color[i][j]
            for pos, bins in enumerate(interval):
              if R >= bins[0] and R < bins[1]:
                posR = pos
            for pos, bins in enumerate(interval):
              if B >= bins[0] and B < bins[1]:
                posB = pos
            for pos, bins in enumerate(interval):
              if G >= bins[0] and G < bins[1]:
                posG = pos
            hists[posR][posG][posB] += 1


    # normalize the histogram such that its integral (sum) is equal 1
    # your code here

    hists = hists.reshape(hists.size)
    return hists

#  compute joint histogram for r/g values
#  note that r/g values should be in the range [0, 1];
#  histogram should be normalized so that sum of all values equals 1
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
def rg_hist(img_color, num_bins):

    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'
  
    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # your code here

    hists = hists.reshape(hists.size)
    return hists


#  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
#  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
#  histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input grayvalue image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  note: you can use the function gaussderiv.m from the filter exercise.
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # compute the first derivatives
    # ...

    # quantize derivatives to "num_bins" number of values
    # ...

    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    # ...
    
    hists = hists.reshape(hists.size)
    return hists

def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
  if dist_name == 'grayvalue':
    return normalized_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rgb':
    return rgb_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rg':
    return rg_hist(img1_gray, num_bins_gray)
  elif dist_name == 'dxdy':
    return dxdy_hist(img1_gray, num_bins_gray)
  else:
    assert 'unknown distance: %s'%dist_name
  

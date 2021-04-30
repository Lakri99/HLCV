import numpy as np
import math

# 
# compute chi2 distance between x and y
#
def dist_chi2(x,y):
  # your code here
  if x + y:
    return np.sum(np.divide((x - y) ** 2, x + y ))
  else:
    return np.sum(np.divide((x - y) ** 2, 1 ))

# 
# compute l2 distance between x and y
#
def dist_l2(x,y):
  # your code here
  return math.sqrt(np.sum((x - y) ** 2))

# 
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similart histograms
#
def dist_intersect(x,y):
  # your code here
  return (1- np.sum(np.minimum(x,y)))

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert 'unknown distance: %s'%dist_name
  





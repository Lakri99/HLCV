import numpy as np
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module
from sklearn.preprocessing import normalize

# 
# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image
#
def plot_rpc(D, plot_color):
  recall = []
  precision = []
  # Total query images
  total_imgs = D.shape[1] 
  #model imagees
  num_images = D.shape[0]
  assert(D.shape[0] == D.shape[1])

  labels = np.diag([1]*num_images)
  
   #normalize [0,1]
  d = (D - np.min(D))/np.ptp(D)

  d = D.reshape(D.size)
  l = labels.reshape(labels.size)

  sortidx = d.argsort()

  d = d[sortidx]
  l = l[sortidx]

  # Threshold range
  tau = np.linspace(0, 1, num=1000)
  
  for i in range(len(tau)):
    tp = 0 #true positive
    fp = 0 #false positive
    fn = 0 #false negative
    tn = 0 #true negative
    for idx in range(len(d)):
      if(d[idx] <= tau[i]): # check on d < tau
        if (l[idx]==0): # incorrect match
          fp+=1
        else: # correct match
          tp+=1
      elif (l[idx]==1): # d > tau and correct match
        fn+=1 
      else:
        tn+=1  # d > tau and incorrect match

    retrieved = tp + fp
    relevant = tp + fn
    if retrieved !=0 and relevant!=0:
      prec = tp/(tp+fp)
      precision.append(prec)
      rec = tp/(tp+fn)
      recall.append(rec)
   
  plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')


def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):

  assert len(plot_colors) == len(dist_types)
  # For rgb hist num_bins is float as (eval_num_bins / 2) results in a float value
  if hist_type == 'rgb':
    num_bins = int(num_bins)

  for idx in range( len(dist_types) ):
    
    [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)
    
    plot_rpc(D, plot_colors[idx])
    

  plt.axis([0, 1, 0, 1]);
  plt.xlabel('1 - precision');
  plt.ylabel('recall');

  # legend(dist_types, 'Location', 'Best')

  plt.legend( dist_types, loc='best')
  






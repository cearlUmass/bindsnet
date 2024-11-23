import torch
from Reservoir import Reservoir
from Memory import sparsify, assign_inhibition
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt

def create_reservoir(exc_size, inh_size, num_grid_cells, gc_multiples,
                     hyper_params, plot=False, save=True):
  print("Creating Reservoir...")

  ## Create synaptic weights ##
  in_size = num_grid_cells * gc_multiples
  w_in_exc = torch.rand(in_size, exc_size)    # Initialize weights
  w_in_inh = torch.rand(in_size, inh_size)
  w_exc_exc = torch.rand(exc_size, exc_size)
  w_exc_inh = torch.rand(exc_size, inh_size)
  w_inh_exc = -torch.rand(inh_size, exc_size)
  w_inh_inh = -torch.rand(inh_size, inh_size)
  w_in_exc = sparsify(w_in_exc, 0.85)   # 0 x% of weights
  w_in_inh = sparsify(w_in_inh, 0.85)
  w_exc_exc = sparsify(w_exc_exc, 0.8)
  w_exc_inh = sparsify(w_exc_inh, 0.5)
  w_inh_exc = sparsify(w_inh_exc, 0.7)
  w_inh_inh = sparsify(w_inh_inh, 0.85)
  res = Reservoir(in_size, exc_size, inh_size, hyper_params,
                  w_in_exc, w_in_inh, w_exc_exc, w_exc_inh, w_inh_exc, w_inh_inh)

  ## Save ##
  if save:
    with open('Data/reservoir_module.pkl', 'wb') as f:
      pkl.dump(res, f)

  return res
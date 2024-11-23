import pickle as pkl
import numpy as np
import torch
from matplotlib import pyplot as plt

def forward_reservoir(exc_size, inh_size, sim_time, spike_trains, labels, reservoir, plot=False, save=True):
  print("Forwarding memories...")

  ## Load memory module and memory keys ##
  # if save:
  #   with open('Data/reservoir_module.pkl', 'rb') as f:
  #     res_module = pkl.load(f)
  #   with open('Data/grid_cell_spk_trains.pkl', 'rb') as f:
  #     memory_keys, labels = pkl.load(f)

  ## Recall memories ##
  recalled_memories = np.zeros((len(spike_trains), sim_time, exc_size + inh_size))
  recalled_memories_sorted = {}
  for i, (key, label) in enumerate(zip(spike_trains, labels)):
    exc_spikes, inh_spikes = reservoir.recall(torch.tensor(key.reshape(sim_time, -1)), sim_time=sim_time)  # Recall the sample
    all_spikes = torch.cat((exc_spikes, inh_spikes), dim=2).squeeze()
    recalled_memories[i] = all_spikes  # Store the recalled memory
    label = tuple(label.round())
    if label not in recalled_memories_sorted:
      recalled_memories_sorted[label] = [all_spikes]
    else:
      recalled_memories_sorted[label].append(all_spikes)

  ## Save recalled memories ##
  if save:
    with open('Data/recalled_memories.pkl', 'wb') as f:
      pkl.dump((recalled_memories, labels), f)
    with open('Data/recalled_memories_sorted.pkl', 'wb') as f:
      pkl.dump(recalled_memories_sorted, f)

  return recalled_memories, labels, recalled_memories_sorted
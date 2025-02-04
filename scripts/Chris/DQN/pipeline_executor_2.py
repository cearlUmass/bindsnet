import matplotlib.pyplot as plt
import numpy as np
import torch

from Grid_Cells import GC_Module


# Generate grid cell activity for each coordinate in the environment
from Reservoir import Reservoir


def grid_cell_activity_generator(maze_size, gc_m: GC_Module):
  # Generate the spike activity for each coordinate in the environment
  x_range, y_range = maze_size
  activity = np.zeros((x_range, y_range, gc_m.n_cells))
  for i in range(x_range):
    for j in range(y_range):
      pos = (i, j)
      a = gc_m.activity(pos)
      activity[i, j] = a
  return activity


# Convert grid cell activity to spike trains
# Return spike trains of shape (x, y, n_cells, sim_time)
def spike_train_generator(gc_activity: np.array, sim_time, max_firing_rates):
  # Note: gc_activity values in range of values [0, 1]
  time_denominator = 1000  # working in ms
  x_range, y_range, n_cells = gc_activity.shape
  spike_trains = np.zeros((*gc_activity.shape, sim_time))
  for i in range(x_range):
    for j in range(y_range):
      for k in range(n_cells):
        activity = gc_activity[i, j, k]   # in range [0, 1]
        max_freq = max_firing_rates[k]    # max firing rate for this grid cell
        spike_rate = activity * max_freq / time_denominator  # spike rate per ms
        spike_train = np.zeros(sim_time)
        if spike_rate != 0:
          step_size = int(1 / spike_rate) # number of ms between spikes
          spike_train[::step_size] = 1
        spike_trains[i, j, k] = spike_train
  return spike_trains


# Calculate the diversity of the grid cell spike trains
# Diversity = avg. difference in # of spikes per grid cell between all pairs of coordinates
def diversity(spike_trains: np.array):
  # spike_trains is a 3D numpy array of shape (x, y, n_cells, sim_time)
  x_range, y_range, n_cells, sim_time = spike_trains.shape
  correlations = np.zeros((x_range*y_range, x_range*y_range))
  for i in range(x_range):
    for j in range(y_range):
      for n in range(x_range):
        for m in range(y_range):
          s1 = spike_trains[i, j].sum(axis=1)  # total number of spikes for each cell
          s2 = spike_trains[n, m].sum(axis=1)
          corr = np.sum(np.abs(s1 - s2))       # Pairwise difference between spike trains
          corr /= n_cells                      # Normalize by number of cells (avg. difference in spikes)
          idx = (i*x_range + j, n*x_range + m)
          correlations[idx] = corr
  return correlations


def generate_weights(in_size, out_size, sparsity):
  w = np.random.uniform(0, 1, (in_size, out_size))
  sparsity_mask = np.random.choice([0, 1], w.shape, p=[1-sparsity, sparsity])
  w *= sparsity_mask
  return w


def run(parameters: dict):
  ## Run Parameters ##
  PLOT = parameters['plot']
  MAZE_SIZE = parameters['maze_size']
  NUM_CELLS = parameters['num_cells']
  X_OFFSETS = parameters['x_offsets']
  Y_OFFSETS = parameters['y_offsets']
  ROTATIONS = parameters['rotations']
  SCALES = parameters['scales']
  SHARPNESSES = parameters['sharpness']
  EXC_SIZE = parameters['exc_size']
  INH_SIZE = parameters['inh_size']
  HYPERPARAMS = parameters['hyperparams']
  SPARSITIES = parameters['sparsities']

  ## Grid Cell activity generator ##
  gc_m = GC_Module(NUM_CELLS, X_OFFSETS, Y_OFFSETS, ROTATIONS, SCALES, SHARPNESSES)
  gc_activity = grid_cell_activity_generator(MAZE_SIZE, gc_m)

  ## Convert Grid Cell activity to spike trains ##
  gc_spike_trains = spike_train_generator(gc_activity, sim_time=1000, max_firing_rates=gc_m.max_firing_rates)

  # Plot the grid cell spike trains
  # Also calculate the diversity in grid cell activity
  if PLOT:
    # Spike trains + Firing Peaks
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(MAZE_SIZE[0], MAZE_SIZE[1]*2)
    fp_ax = fig.add_subplot(gs[:, MAZE_SIZE[1]:])   # firing peak axis
    for i in range(MAZE_SIZE[0]):
      for j in range(MAZE_SIZE[1]):
        fp_ax.plot(i, j, '+', color='black')
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(gc_spike_trains[i, j], aspect='auto', cmap='binary', interpolation=None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({i}, {j})")
    gc_m.plot_peaks([-1, MAZE_SIZE[0]], [-1, MAZE_SIZE[1]], fig=fig, ax=fp_ax)
    fp_ax.set_title("Grid Cell Firing Peaks")
    # fig.tight_layout()

    # Diversity
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    d = diversity(gc_spike_trains)
    im = ax.imshow(d, cmap='viridis')
    fig.colorbar(im)
    anti_diag = ~np.eye(d.shape[0], dtype=bool)  # Anti-diagonal elements
    avg_div = np.mean(np.abs(d[anti_diag]))    # Average diversity without diagonal
    ax.set_title(f"Avg. Diversity: {avg_div:.2f}")
    plt.show()

  ## Push spike trains through association area ##
  w_in_exc = generate_weights(NUM_CELLS, EXC_SIZE, SPARSITIES['in_exc'])
  w_in_inh = generate_weights(NUM_CELLS, INH_SIZE, SPARSITIES['in_inh'])
  w_exc_exc = generate_weights(EXC_SIZE, EXC_SIZE, SPARSITIES['exc_exc'])
  w_exc_inh = generate_weights(EXC_SIZE, INH_SIZE, SPARSITIES['exc_inh'])
  w_inh_exc = -generate_weights(INH_SIZE, EXC_SIZE, SPARSITIES['inh_exc'])
  w_inh_inh = -generate_weights(INH_SIZE, INH_SIZE, SPARSITIES['inh_inh'])
  reservoir = Reservoir(
             in_size=NUM_CELLS,
             exc_size=EXC_SIZE,
             inh_size=INH_SIZE,
             w_in_exc=w_in_exc,
             w_in_inh=w_in_inh,
             w_exc_exc=w_exc_exc,
             w_exc_inh=w_exc_inh,
             w_inh_exc=w_inh_exc,
             w_inh_inh=w_inh_inh,
             hyper_params=HYPERPARAMS,)
  res_spike_trains = torch.zeros(MAZE_SIZE[0], MAZE_SIZE[1], EXC_SIZE+INH_SIZE, 1000)
  for i in range(MAZE_SIZE[0]):
    for j in range(MAZE_SIZE[1]):
      exc_spikes, inh_spikes = reservoir.get_spikes(gc_spike_trains[i, j], sim_time=1000)  # Run for 1 second
      res_spike_trains[i, j] = torch.concat((exc_spikes, inh_spikes), dim=2).squeeze(1).T  # (time, exc+inh)

  # Plot reservoir spike trains
  # Also calculate the diversity in reservoir activity
  if PLOT:
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(MAZE_SIZE[0], MAZE_SIZE[1]*2)
    div_ax = fig.add_subplot(gs[:, MAZE_SIZE[1]:])   # firing peak axis
    for i in range(MAZE_SIZE[0]):
      for j in range(MAZE_SIZE[1]):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(res_spike_trains[i, j], aspect='auto', cmap='binary', interpolation=None)
        ax.set_xticks([])
        ax.set_yticks([])
    # Diversity
    d = diversity(res_spike_trains.numpy())
    im = div_ax.imshow(d, cmap='viridis')
    fig.colorbar(im)
    anti_diag = ~np.eye(d.shape[0], dtype=bool)  # Anti-diagonal elements
    avg_div = np.mean(np.abs(d[anti_diag]))  # Average diversity without diagonal
    div_ax.set_title(f"Avg. Diversity: {avg_div:.2f}")
    plt.show()

if __name__ == '__main__':
  NUM_CELLS = 25
  p = {
    'plot': True,
    'maze_size': (3, 3),
    'num_cells': NUM_CELLS,
    'x_offsets': np.random.uniform(-1, 1, NUM_CELLS),
    'y_offsets': np.random.uniform(-1, 1, NUM_CELLS),
    'rotations': np.random.uniform(-np.pi, np.pi, NUM_CELLS),
    'scales': np.random.uniform(0.5, 1.5, NUM_CELLS),
    'sharpness': np.ones(NUM_CELLS),    # Should *not* go below 1
    'exc_size': 100,
    'inh_size': 25,
    'hyperparams': {
      "exc_refrac": 5,
      "exc_reset": -64,
      "exc_tc_decay": 10_000,
      "exc_tc_theta_decay": 10_000,
      "exc_theta_plus": 0,
      "exc_thresh": -60,
      "inh_refrac": 5,
      "inh_reset": -64,
      "inh_tc_decay": 10_000,
      "inh_tc_theta_decay": 10_000,
      "inh_theta_plus": 0,
      "inh_thresh": -60,
      "refrac_out": 5,
      "reset_out": -64,
      "tc_decay_out": 10_000,
      "tc_theta_decay_out": 10_000,
      "theta_plus_out": 0,
      "thresh_out": -60,
    },
    'sparsities': {
      'in_exc': 0.9,
      'in_inh': 0.9,
      'exc_exc': 0.9,
      'exc_inh': 0.9,
      'inh_exc': 0.9,
      'inh_inh': 0.9,
    },
  }
  run(p)
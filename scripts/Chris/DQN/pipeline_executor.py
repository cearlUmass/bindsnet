import numpy as np
import pickle as pkl

import torch

from Environment import Grid_Cell_Maze_Environment
from train_DQN import train_DQN
from STDP_RL import train_STDP_RL
from sample_generator import sample_generator
from spike_train_generator import spike_train_generator
from create_reservoir import create_reservoir
from recall_reservoir import forward_reservoir
from recalled_mem_preprocessing import recalled_mem_preprocessing
from classify_recalls import classify_recalls

if __name__ == '__main__':
  ## Constants ##
  WIDTH = 5
  HEIGHT = 5
  SAMPLES_PER_POS = 1
  NOISE = 0.1   # Noise in sampling
  NUM_CELLS = 25
  X_RANGE = (0, WIDTH)
  Y_RANGE = (0, HEIGHT)
  SIM_TIME = 50
  MAX_SPIKE_FREQ = 0.8
  GC_MULTIPLES = 1
  EXC_SIZE = 500
  INH_SIZE = 100
  STORE_SAMPLES = 0
  exc_hyper_params = {
      'thresh_exc': -55,
      'theta_plus_exc': 0,
      'refrac_exc': 1,
      'reset_exc': -65,
      'tc_theta_decay_exc': 500,
      'tc_decay_exc': 30,
    }
  inh_hyper_params = {
    'thresh_inh': -55,
    'theta_plus_inh': 0,
    'refrac_inh': 1,
    'reset_inh': -65,
    'tc_theta_decay_inh': 500,
    'tc_decay_inh': 30,
  }
  PLOT = True

  ## Sample Generation ##
  # x_offsets = np.random.uniform(-1, 1, NUM_CELLS)
  # y_offsets = np.random.uniform(-1, 1, NUM_CELLS)
  # offsets = list(zip(x_offsets, y_offsets))           # Grid Cell x & y offsets
  # scales = [np.random.uniform(0.5, 3) for i in range(NUM_CELLS)]   # Dist. between Grid Cell peaks
  # vars = [.85] * NUM_CELLS              # Width of grid cell activity
  # samples, labels, sorted_samples = sample_generator(scales, offsets, vars, X_RANGE, Y_RANGE, SAMPLES_PER_POS,
  #                                                    noise=NOISE, padding=1, plot=PLOT)
  #
  # ## Spike Train Generation ##
  # spike_trains, labels, sorted_spike_trains = spike_train_generator(SIM_TIME, GC_MULTIPLES, MAX_SPIKE_FREQ)

  # Create Reservoir ##
  # hyper_params = exc_hyper_params | inh_hyper_params
  # create_reservoir(EXC_SIZE, INH_SIZE, STORE_SAMPLES, NUM_CELLS, GC_MULTIPLES, SIM_TIME, hyper_params, PLOT)
  #
  # # ## Association (Recall) ##
  # forward_reservoir(EXC_SIZE, INH_SIZE, SIM_TIME, PLOT)

  ## Preprocess Recalls ##
  # recalled_mem_preprocessing(WIDTH, HEIGHT, PLOT)

  ## Train STDP-RL ##
  EPS_START = 0.75
  EPS_END = 0.05
  DECAY_INTENSITY = 3  # higher
  # GAMMA = 0.99
  MAX_STEPS_PER_EP = 50
  MAX_TOTAL_STEPS = 5000
  INPUT_SIZE = EXC_SIZE # + INH_SIZE
  MOTOR_POP_SIZE = 50
  OUT_SIZE = 4 * MOTOR_POP_SIZE
  A_PLUS = 1
  A_MINUS = 0
  TC_E_TRACE = 10
  SIM_TIME = 50
  ENV_TRACE_LENGTH = 10
  LR = 0.01
  GAMMA = 0.7
  out_hyperparams = {
    'thresh_out': -60,
    'theta_plus_out': 0,
    'refrac_out': 1,
    'reset_out': -65,
    'tc_theta_decay_out': 1000,
    'tc_decay_out': 30,
  }
  train_STDP_RL(HEIGHT, WIDTH, MAX_TOTAL_STEPS, MAX_STEPS_PER_EP, EPS_START, EPS_END, DECAY_INTENSITY,
                INPUT_SIZE, OUT_SIZE, MOTOR_POP_SIZE, SIM_TIME, out_hyperparams, ENV_TRACE_LENGTH,
                A_PLUS, A_MINUS, TC_E_TRACE, LR, GAMMA, device='cpu', plot=PLOT)

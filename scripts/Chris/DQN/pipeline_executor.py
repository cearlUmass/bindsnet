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

def run(parameters: dict):
  # If you want plots
  PLOT = False

  ## Model Constants ##
  WIDTH = parameters['WIDTH']
  HEIGHT = parameters['HEIGHT']
  SAMPLES_PER_POS = parameters['SAMPLES_PER_POS']
  NOISE = parameters['NOISE']
  NUM_CELLS = parameters['NUM_CELLS']
  SIM_TIME = parameters['SIM_TIME']
  MAX_SPIKE_FREQ = parameters['MAX_SPIKE_FREQ']
  GC_MULTIPLES = parameters['GC_MULTIPLES']
  EXC_SIZE = parameters['EXC_SIZE']
  INH_SIZE = parameters['INH_SIZE']
  exc_hyper_params = {
    'thresh_exc': parameters['exc_thresh'],
    'theta_plus_exc': parameters['exc_theta_plus'],
    'refrac_exc': parameters['exc_refrac'],
    'reset_exc': parameters['exc_reset'],
    'tc_theta_decay_exc': parameters['exc_tc_theta_decay'],
    'tc_decay_exc': parameters['exc_tc_decay'],
  }
  inh_hyper_params = {
    'thresh_inh': parameters['inh_thresh'],
    'theta_plus_inh': parameters['inh_theta_plus'],
    'refrac_inh': parameters['inh_refrac'],
    'reset_inh': parameters['inh_reset'],
    'tc_theta_decay_inh': parameters['inh_tc_theta_decay'],
    'tc_decay_inh': parameters['inh_tc_decay'],
  }

  ## Train Constants ##
  EPS_START = parameters['EPS_START']
  EPS_END = parameters['EPS_END']
  DECAY_INTENSITY = parameters['DECAY_INTENSITY']
  MAX_STEPS_PER_EP = parameters['MAX_STEPS_PER_EP']
  MAX_TOTAL_STEPS = parameters['MAX_TOTAL_STEPS']
  MOTOR_POP_SIZE = parameters['MOTOR_POP_SIZE']
  OUT_SIZE = parameters['OUT_SIZE']
  ENV_TRACE_LENGTH = parameters['ENV_TRACE_LENGTH']
  LR = parameters['LR']
  GAMMA = parameters['GAMMA']
  out_hyperparams = {
    'thresh_out': parameters['thresh_out'],
    'theta_plus_out': parameters['theta_plus_out'],
    'refrac_out': parameters['refrac_out'],
    'reset_out': parameters['reset_out'],
    'tc_theta_decay_out': parameters['tc_theta_decay_out'],
    'tc_decay_out': parameters['tc_decay_out'],
  }

  ## Sample Generation ##
  x_offsets = np.random.uniform(-1, 1, NUM_CELLS)
  y_offsets = np.random.uniform(-1, 1, NUM_CELLS)
  offsets = list(zip(x_offsets, y_offsets))  # Grid Cell x & y offsets
  scales = [np.random.uniform(0.5, 3) for i in range(NUM_CELLS)]  # Dist. between Grid Cell peaks
  vars = [.25] * NUM_CELLS  # Width of grid cell activity
  sample_generator(scales, offsets, vars, (0, WIDTH), (0, HEIGHT), SAMPLES_PER_POS,
                                                     noise=NOISE, padding=1, plot=PLOT)

  ## Spike Train Generation ##
  spike_train_generator(SIM_TIME, GC_MULTIPLES, MAX_SPIKE_FREQ)

  ## Create association area ##
  hyper_params = exc_hyper_params | inh_hyper_params
  create_reservoir(EXC_SIZE, INH_SIZE, NUM_CELLS, GC_MULTIPLES, hyper_params, PLOT)

  ## Pass Grid-Cell spike train through association area ##
  forward_reservoir(EXC_SIZE, INH_SIZE, SIM_TIME, PLOT)

  ## Train model w/ STDP-RL ##
  score = train_STDP_RL(HEIGHT, WIDTH, MAX_TOTAL_STEPS, MAX_STEPS_PER_EP, EPS_START, EPS_END,
                DECAY_INTENSITY, EXC_SIZE, OUT_SIZE, MOTOR_POP_SIZE, SIM_TIME, out_hyperparams, ENV_TRACE_LENGTH,
                LR, GAMMA, device='cpu', plot=PLOT)
  print(f"Score: {score}")
  return score

if __name__ == '__main__':
  p = {
    # Model Constants
    'WIDTH': 5,                 # Width of the grid
    'HEIGHT': 5,                # Height of the grid
    'SAMPLES_PER_POS': 1,       # Number of samples per position
    'NOISE': 0.1,               # Noise in sampling
    'NUM_CELLS': 35,            # Number of grid cells
    'SIM_TIME': 50,             # How long (ms) to run model per step
    'MAX_SPIKE_FREQ': 0.8,      # Maximum spike frequency for Grid Cells
    'GC_MULTIPLES': 1,          # How many repeats of Grid Cells
    'EXC_SIZE': 1000,           # Number of excitatory neurons
    'INH_SIZE': 250,            # Number of inhibitory neurons

    # Excitatory neuron hyperparameters
    'exc_thresh': -55,          # Firing threshold
    'exc_theta_plus': 0,        # Increase in refractory
    'exc_refrac': 1,            # base refractory
    'exc_reset': -65,           # Membrane reset
    'exc_tc_theta_decay': 500,  # Refractory rate of decay (bigger = slower decay)
    'exc_tc_decay': 30,         # Membrane potential rate of decay (bigger = slower decay)

    # Inhibitory neuron hyperparameters
    'inh_thresh': -55,          # Firing threshold
    'inh_theta_plus': 0,        # Increase in refractory
    'inh_refrac': 1,            # base refractory
    'inh_reset': -65,           # Membrane reset
    'inh_tc_theta_decay': 500,  # Refractory rate of decay (bigger = slower decay)
    'inh_tc_decay': 30,         # Membrane potential rate of decay (bigger = slower decay)

    # Output neuron hyperparameters
    'thresh_out': -60,          # Firing threshold
    'theta_plus_out': 0,        # Increase in refractory
    'refrac_out': 1,            # base refractory
    'reset_out': -65,           # Membrane reset
    'tc_theta_decay_out': 1000, # Refractory rate of decay (bigger = slower decay)
    'tc_decay_out': 30,         # Membrane potential rate of decay (bigger = slower decay

    # Training Constants
    'EPS_START': 0.95,          # Epsilon starting val
    'EPS_END': 0,               # Epsilon final val
    'DECAY_INTENSITY': 3,       # Rate of exponential decay for epsilon
    'MAX_STEPS_PER_EP': 100,    # Max steps per episode
    'MAX_TOTAL_STEPS': 2000,    # Max total steps
    'MOTOR_POP_SIZE': 50,       # Number of motor neurons per action
    'OUT_SIZE': 4 * 50,         # Motor-Output size (Should be multiple of MOTOR_POP_SIZE)
    'ENV_TRACE_LENGTH': 8,      # Length of the environment trace
    'LR': 0.001,                # Weight Learning rate
    'GAMMA': 0.7,               # Q-Learning Discount factor
  }
  r = {
    'NUM_CELLS': [35, 50],            # Integer
    'EXC_SIZE': [1000, 2000],         # Integer
    'INH_SIZE': [250, 500],           # Integer
    'exc_thresh': [-55, -50],         # Float/Integer
    'exc_theta_plus': [0, 5],         # Float/Integer
    'exc_refrac': [1, 5],             # Float/Integer
    'exc_reset': [-65, -60],          # Float/Integer
    'exc_tc_theta_decay': [500, 1000],  # Float/Integer
    'exc_tc_decay': [30, 50],         # Float/Integer
    'inh_thresh': [-55, -50],         # Float/Integer
    'inh_theta_plus': [0, 5],         # Float/Integer
    'inh_refrac': [1, 5],             # Float/Integer
    'inh_reset': [-65, -60],          # Float/Integer
    'inh_tc_theta_decay': [500, 1000],# Float/Integer
    'inh_tc_decay': [30, 50],         # Float/Integer
    'thresh_out': [-60, -55],         # Float/Integer
    'theta_plus_out': [0, 5],         # Float/Integer
    'refrac_out': [1, 5],             # Float/Integer
    'reset_out': [-65, -60],          # Float/Integer
    'tc_theta_decay_out': [1000, 2000], # Float/Integer
    'tc_decay_out': [30, 50],         # Float/Integer
    'DECAY_INTENSITY': [3, 5],        # Float/Integer
    'MAX_STEPS_PER_EP': [100, 200],   # Integer
    'MAX_TOTAL_STEPS': [2000, 5000],  # Integer
    'MOTOR_POP_SIZE': [50, 100],      # Integer
    'LR': [0.001, 0.01],              # Float
    'GAMMA': [0.7, 0.9],              # Float
  }
  run(p)

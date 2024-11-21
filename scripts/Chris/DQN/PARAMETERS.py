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
    'exc_thresh': -55,          #  Firing threshold
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
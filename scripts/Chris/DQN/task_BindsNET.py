import torch
import os, sys, yaml, subprocess, signal, argparse, time
from datetime import datetime


global args
global taskPath

import numpy as np

from pipeline_executor import *

# Set random seed for reproducibility
torch.manual_seed(42)

# Set the device (GPU if available, else CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')  # Use the first GPU


def run_task(param, args):
    global taskPath 

    # show I am alive!
                                                                                                    
    file = os.path.abspath(taskPath.replace('"','') + '/' + str(args.agent_test_num) + '.live.lock')
    try:
        with open(file, 'wt') as f:
            f.write('.')
    except:
        print('Problem creating agent ' + str(args.agent_idx) + ', task number' + str(args.agent_test_num) + ' lock file: ' + file)
        return

    import time
    print(f'---- Start Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    start_time = time.time()

    #-----------------------------------------------------
    outP = dict()

    import random

    saveSeed = random.getstate()

    p = {
    # Model Parameters
    'NOISE': 0.1,               # Noise in sampling
    'NUM_CELLS': 35,            # Number of grid cells
    'SIM_TIME': 50,             # How long (ms) to run model per step
    'MAX_SPIKE_FREQ': 0.8,      # Maximum spike frequency for Grid Cells
    'EXC_SIZE': 1000,           # Number of excitatory neurons
    'INH_SIZE': 250,            # Number of inhibitory neurons
    'GRID_CELL_RADIUS': .25,    # Width of grid cell activity
    'GRID_CELL_SCALE': 3,       # Dist. between Grid Cell peaks

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

    # Training Parameters
    'DECAY_INTENSITY': 3,       # Rate of exponential decay for epsilon
    'MAX_TOTAL_STEPS': 2000,    # Max total steps during training
    'MOTOR_POP_SIZE': 50,       # Number of motor neurons per action
    'LR': 0.001,                # Weight Learning rate
    'GAMMA': 0.7,               # Q-Learning Discount factor
  }
    
    c = {
    
        # Model Parameters
        'SIM_TIME': 50,             # How long (ms) to run model per step
        'MAX_SPIKE_FREQ': 0.8,      # Maximum spike frequency for Grid Cells

        # Model Constants
        'WIDTH': 5,                 # Width of the grid
        'HEIGHT': 5,                # Height of the grid
        'SAMPLES_PER_POS': 1,       # Number of samples per position
        'NOISE': 0.1,               # Noise in sampling
        'GC_MULTIPLES': 1,          # How many repeats of Grid Cells

        # Training Constants
        'EPS_START': 0.95,          # Epsilon starting val
        'EPS_END': 0,               # Epsilon final val
        'MAX_STEPS_PER_EP': 100,    # Max steps per episode
        # 'OUT_SIZE': 4 * p['MOTOR_POP_SIZE'], # Motor-Output population size
        'OUT_SIZE': 4 * param['param']['MOTOR_POP_SIZE'], # Motor-Output population size
        'ENV_TRACE_LENGTH': 8,      # Length of the environment trace
        'ENV_PATH': os.path.dirname(os.path.abspath(__file__))+'/Env/' + 'env.pkl', # Path to the environment
    }

    # Instantiate the model
    model = run(param['param'] | c)

    outP['fitnessScore'] = model
    # outP['Target'] = target_copy

    if np.isnan(model):
        outP['fitnessScore'] = None
    else:
        outP['fitnessScore'] = float(str(outP['fitnessScore']))

    del model
        
    
    random.setstate(saveSeed)
    
    print(outP)
    #-----------------------------------------------------

    print(f'---- End Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    print(f'---- Task Total Time ----- {(time.time() - start_time)} sec', flush=True)

    # delete alive file and terminate job
    os.system('rm ' + os.path.abspath(taskPath.replace('"','') + '/' + str(args.agent_test_num) + '.live.lock'))

    return outP
    
    
def main(args):
    global taskPath 

    taskPath = args.path + "/running/" + str(args.agent_idx)
    os.makedirs(taskPath, exist_ok=True)
    
    # load the gene from save tasks
    with open(args.path + "/running/" + str(args.agent_idx) + '.newMember.yaml', 'rt') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    
    # try:
    for gene_num in range(args.total_testsGene_with_same_agent):
        fitness = []
        for test in range(args.total_test_with_same_param):
            #-----------------------
            outP = run_task(param[gene_num], args)
            #-----------------------
            if outP['fitnessScore'] is not None:
                fitness.append(outP['fitnessScore'])    

        if len(fitness) > 0:
            outP['fitnessScore'] = fitness

        if not outP is None:
            for k,v in outP.items():
                param[gene_num][k] = v

    # except Exception as e:
        # print('Problem with the task number: ' + str(args.agent_test_num) + ' for agent: ' + str(args.agent_idx), flush=True)
        # print(e, flush=True)
    
    # write resoults
    with open(taskPath + '/' + str(args.agent_test_num)  + '.resoult.yaml', 'wt') as f:
        yaml.dump(param, f)
        
    if args.parallel or args.slurm:   
        callBack(args)
    
    # # delete the gene file
    # os.system('rm ' + args.path + "/running/" + str(args.agent_idx) + '.newMember.yaml')
    
    print('Done Task: ' + str(args.agent_test_num) + ' for agent: ' + str(args.agent_idx), flush=True)
    return

def signal_handler_SIGTERM(sig, frame):
    global args
    global taskPath 
    os.system('rm ' + os.path.abspath(taskPath.replace('"','') + '/' + str(args.agent_test_num) + '.live.lock'))
    runMyself(args=argparse._copy_items(args),)
    print('got a SIGTERM!!')
    
    ##### consider deleteing lok on log file for best gene
        
    print('Done signal_handler')
    sys.exit(0)
signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
signal.signal(signal.SIGUSR1, signal_handler_SIGTERM)
signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
signal.signal(signal.SIGINT, signal_handler_SIGTERM)

def runMyself(args,):
    # tmp_s = args.method[0] if len(args.method) == 1 else '\"'+args.method[0]+";"+args.method[1]+'\"' 
    run_process = [ 
            __file__,
            "--callBackScript", '"' + args.callBackScript + '"' if args.parallel else args.callBackScript.replace('"',''),
            "--path", '"' + args.path + '"' if args.parallel else args.path.replace('"',''),
            '--agent_idx', str(args.agent_idx),
            "--agent_test_num", str(args.agent_test_num),
            "--agent_counter", str(args.agent_counter),
            "--total_test_with_same_param", str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'True' if args.parallel else 'False',
            # '--fitness', str(args.fitness),
            # '--deversity', str(args.deversity),
            '--restart_window', str(args.restart_window),
            # '--method', tmp_s,
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', args.paramFile,
            ]

    if args.slurm:
        temp_file = 'TaskRunSelf_Agent_idx-' + str(args.agent_idx) + '_Task-' + str(args.agent_test_num) + datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")+'.sh'
        with open(temp_file, 'wt') as f:
            f.write(str(run_process).replace('[','').replace(']','').replace('\', \'',' ').replace('\'',''))
        if args.gpu:
            os.system('sbatch ' + __file__.split('.')[0] + '.gpu.sh ' + temp_file)
        else:
            os.system('sbatch ' + __file__.split('.')[0] + '.sh ' + temp_file)
    elif args.parallel:
        run_process = str(['python3'] + run_process).replace('[','').replace(']','').replace('\'','').replace(',', ' ')
        subprocess.Popen(run_process, shell=True)        

def callBack(args,):
    if args.callBackScript is None: 
        return
    # tmp_s = args.method[0] if len(args.method) == 1 else '\"'+args.method[0]+";"+args.method[1]+'\"'
    run_process = [ 
            '"' + args.callBackScript + '"' if args.parallel else args.callBackScript.replace('"',''),
            '--part', '2',
            '--taskFilename', '"' + __file__ + '"' if args.parallel else __file__,
            "--path", '"' + args.path + '"' if args.parallel else args.path,
            '--agent_idx', str(args.agent_idx),
            "--agent_test_num", str(args.agent_test_num),
            "--agent_counter", str(args.agent_counter),
            "--total_test_with_same_param", str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'True' if args.parallel else 'False',
            # '--fitness', str(args.fitness),
            # '--deversity', str(args.deversity),
            '--restart_window', str(args.restart_window),
            # '--method', tmp_s,
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', args.paramFile,
            ]

    if args.slurm:
        temp_file = 'Task_CallBack_Agent_idx-' + str(args.agent_idx) + '_Task-' + str(args.agent_test_num) + datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")+'.sh'
        with open(temp_file, 'wt') as f:
            f.write(str(run_process).replace('[','').replace(']','').replace('\', \'',' ').replace('\'',''))
        r = os.system('sbatch ' + args.callBackScript.split('.')[0] + '.sh" ' + temp_file)
        if r != 0:
            r = os.system('sbatch ' + args.callBackScript.split('.')[0] + '.sh ' + temp_file)
    else:
        run_process = str(['python3'] + run_process).replace('[','').replace(']','').replace('\'','').replace(',', ' ')
        subprocess.Popen(run_process, shell=True)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='population')       # pass to the next process
    parser.add_argument("--agent_test_num", type=int, default=-1)       # receved by the child process
    parser.add_argument("--agent_idx", type=int, default=-1)            # modify by the process
    parser.add_argument("--agent_counter", type=str, default='-1.0')        # modified by the next process
    parser.add_argument("--total_test_with_same_param", type=int, default=1) # pass to the next process
    parser.add_argument("--total_testsGene_with_same_agent", type=int, default=1)  # pass to the next
    parser.add_argument("--gpu", type=str, default='False')        # pass to the next process
    parser.add_argument("--slurm", type=str, default='False')      # pass to the next process
    parser.add_argument("--parallel", type=str, default='True')      # NO need to pass to the next process or to myself
    parser.add_argument("--callBackScript", type=str, default=None)
    # parser.add_argument("--fitness", type=float, default=None)
    # parser.add_argument("--deversity", type=float, default=None)
    parser.add_argument("--restart_window", type=int, default=10)
    # parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--evolutionTarget", type=int, default=1)      # 1 = Max, -1 = Min
    parser.add_argument('--paramFile', type=str, default=None)
    
    args = parser.parse_args()
    
    args.gpu = True if args.gpu == 'True' or args.gpu == 'true' else False
    args.slurm = True if args.slurm == 'True' or args.slurm == 'true' else False
    args.parallel = True if args.parallel == 'True' or args.parallel == 'true' else False
    
    # args.method = [item for item in args.method.replace(' ','').replace('"','').split(';')]

    if args.slurm: 
        args.parallel = False


    main(argparse._copy_items(args))

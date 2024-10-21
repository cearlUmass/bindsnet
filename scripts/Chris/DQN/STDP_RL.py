import math
import random
import pickle as pkl
import numpy as np
import torch
from itertools import count

from matplotlib import pyplot as plt

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDP
from scripts.Chris.DQN.Environment import Grid_Cell_Maze_Environment
from scripts.Chris.DQN.Memory import sparsify


class STDP_RL_Model(Network):
  def __init__(self, in_size, out_size, hyper_params, w_in_out, w_out_out, a_plus, a_minus, tc_e_trace,
               learning_rate, gamma, device='cpu'):
    super().__init__()

    ## Layers ##
    input = Input(n=in_size)
    output = AdaptiveLIFNodes(
      n=out_size,
      thresh=hyper_params['thresh_out'],
      theta_plus=hyper_params['theta_plus_out'],
      refrac=hyper_params['refrac_out'],
      reset=hyper_params['reset_out'],
      tc_theta_decay=hyper_params['tc_theta_decay_out'],
      tc_decay=hyper_params['tc_decay_out'],
      traces=True,
    )
    output_monitor = Monitor(output, ["s"], device=device)
    self.output_monitor = output_monitor
    self.add_monitor(output_monitor, name='output_monitor')
    self.add_layer(input, name='input')
    self.add_layer(output, name='output')

    ## Connections ##
    in_out_wfeat = Weight(name='in_out_weight_feature', value=w_in_out)
    in_out_conn = MulticompartmentConnection(
      source=input, target=output,
      device=device, pipeline=[in_out_wfeat],
    )
    out_out_wfeat = Weight(name='out_out_weight_feature', value=w_out_out)
    out_out_conn = MulticompartmentConnection(
      source=output, target=output,
      device=device, pipeline=[out_out_wfeat],
    )
    self.add_connection(in_out_conn, source='input', target='output')
    self.add_connection(out_out_conn, source='output', target='output')
    self.weights = in_out_wfeat

    ## Migrate ##
    self.to(device)

    ## STDP-RL Parameters ##
    self.eligibility_trace = torch.zeros(in_size, out_size, device=device)
    self.eligibility = torch.zeros(in_size, out_size, device=device)
    self.a_plus = a_plus
    self.a_minus = a_minus
    self.tc_e_trace = tc_e_trace
    self.lr = learning_rate

    ## Q-Learning parameters ##
    self.gamma = gamma
    self.q_table = {}

  def STDP_RL(self, update_mod, in_spikes, out_spikes):
    in_activity = in_spikes.squeeze().sum(dim=0)
    out_activity = out_spikes.squeeze().sum(dim=0)
    self.eligibility = torch.outer(in_activity*self.a_plus, out_activity) + \
                  torch.outer(in_activity, out_activity*self.a_minus)

    # Update eligibility trace
    # self.eligibility_trace *= np.exp(-1/self.tc_e_trace)
    # self.eligibility_trace += self.eligibility / self.tc_e_trace

    # Update weights
    self.weights.value += self.eligibility * update_mod # * 0.1

  def update_table(self, state, action, next_state, reward):
    # Preprocess state for simpler indexing
    # Sum spikes -> get sorted indices
    action = action.item()
    if state not in self.q_table:
      self.q_table[state] = np.zeros(4)    # 4 moves
      self.q_table[state][action] = self.lr * reward
      return self.lr * reward
    else:
      next_state_val = 0 if next_state not in self.q_table else max(self.q_table[next_state])
      old_val = self.q_table[state][action]
      self.q_table[state][action] *= (1 - self.lr)  # Decay
      self.q_table[state][action] += self.lr * (reward + self.gamma * next_state_val)  # Update
      delta = self.q_table[state][action] - old_val
      opt_act = np.argmax(self.q_table[state])
      return delta

  def plot(self):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(self.weights.value, cmap='hot', interpolation='nearest')
    fig.colorbar(im, ax=ax1)
    im = ax2.imshow(self.eligibility, cmap='hot', interpolation='nearest')
    fig.colorbar(im, ax=ax2)
    plt.show()


# Select action using epsilon-greedy policy
def select_action(assoc_spikes, sim_time, out_size, eps, model, env):
  motor_pop_size = out_size // env.num_actions

  # Select action from policy net
  if random.random() > eps:
    ## Pass association spikes through model ##
    model.run(inputs={'input': assoc_spikes}, time=sim_time)
    out_spikes = model.output_monitor.get('s')
    action = torch.argmax(out_spikes.reshape(sim_time, env.num_actions, motor_pop_size).sum(0).sum(1))
    model.reset_state_variables()
    return action, out_spikes.squeeze()

  # Select random action (exploration)
  else:
    # Generate artificial out_spikes
    action = np.random.choice(env.num_actions)
    out_spikes = torch.zeros(sim_time, out_size)
    motor_pop_range = (action*motor_pop_size, action*motor_pop_size+motor_pop_size)
    motor_pop_spikes = torch.rand(sim_time, motor_pop_size) < 0.1
    out_spikes[:, motor_pop_range[0]:motor_pop_range[1]] = motor_pop_spikes
    return torch.tensor([action]), out_spikes


def run_episode(env, model, in_size, out_size, max_steps, sim_time, eps=0, learning=False, device='cpu'):
  # Initialize the environment and get its state
  state, coords, _ = env.reset()
  state = state[:, 0:in_size]
  history = []
  model.plot()
  plt.show()
  for t in count():
    action, out_spikes = select_action(state, sim_time, out_size, eps, model, env)
    observation, reward, terminated, next_state_coords, _ = env.step(action)
    # reward = torch.tensor([reward], device=device)

    # Update history
    history.append((state.numpy(), coords, action, reward, out_spikes.numpy()))

    # Break if terminated or max_steps reached
    # (+1 because the first state is not counted as a step)
    if terminated or t >= max_steps:
      model.plot()
      plt.show()
      break
    else:
      next_state = torch.tensor(observation, dtype=torch.bool, device=device)

    # Perform one step of the optimization
    if learning:
      update_mod = model.update_table(coords, action, next_state_coords, reward)  # TODO: Change from coords to spikes
      model.STDP_RL(update_mod, state, out_spikes)

    # Move to the next state
    state = next_state
    coords = next_state_coords
    state = state[:, 0:in_size]

  return history


def record_episode(env, model, in_size, out_size, max_steps, sim_time, eps, learning, device, filename):
  history = run_episode(env, model, in_size, out_size, max_steps, sim_time, eps, learning, device)  # eps = 0 -> no exploration
  env.animate_history(history, filename)
  plt.clf()


def train_STDP_RL(env_width, env_height, max_total_steps, max_steps_per_ep, eps_start,
                  eps_end, decay_intensity, in_size, out_size, sim_time, hyper_params,
                  env_trace_length, a_plus, a_minus, tc_e_trace, learning_rate, gamma, device='cpu',
                  plot=False):

  ## Init model & maze ##
  w_in_out = torch.rand((in_size, out_size))
  w_in_out = sparsify(w_in_out, 0.5)
  w_out_out = -torch.ones((out_size, out_size))*0.1
  for i in range(4):
    w_out_out[i*20:(i+1)*20, i*20:(i+1)*20] = 0
  model = STDP_RL_Model(in_size, out_size, hyper_params, w_in_out, w_out_out,
                        a_plus, a_minus, tc_e_trace, learning_rate, gamma, device)
  env = Grid_Cell_Maze_Environment(width=env_width, height=env_height, trace_length=env_trace_length,
                                   samples_file='Data/recalled_memories_sorted.pkl')

  ## Pre-training recording ##
  # if plot:
  #   record_episode(env, model, in_size, out_size, 25, sim_time, eps=0, learning=False, device=device,
  #                  filename="pre_training.gif")

  ## Training loop ##
  episode_durations = []
  episodes = 0
  total_steps = 0
  print(env.maze)
  while total_steps < max_total_steps:
    eps = eps_end + (eps_start - eps_end) * math.exp(-decay_intensity * total_steps / (max_total_steps))
    history = run_episode(env, model, in_size, out_size, max_steps_per_ep, sim_time, eps=eps, learning=True, device=device)
    total_steps += len(history)
    episode_durations.append(len(history))
    print(f"Episode {episodes} lasted {len(history)} steps, eps = {round(eps, 2)} total steps = {total_steps}")
    episodes += 1

  ## Post-training recording ##
  if plot:
    record_episode(env, model, in_size, out_size, 25, sim_time, eps=0, learning=False, device=device,
                   filename="post_training.gif")

  ## Plot Episodes##
  if plot:
    plt.plot(episode_durations)
    plt.title("Episode durations")
    plt.ylabel("Duration")
    plt.xlabel("Episode")
    plt.show()

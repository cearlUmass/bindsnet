import random
from collections import deque

import numpy as np
from labyrinth.generate import DepthFirstSearchGenerator
from labyrinth.grid import Cell, Direction
from labyrinth.maze import Maze
from labyrinth.solve import MazeSolver
from matplotlib.pyplot import plot as plt
from matplotlib.animation import FuncAnimation

import pickle as pkl
import matplotlib.pyplot as plt
from torch import optim

class Maze_Environment():
  def __init__(self, width, height, trace_length=5):

    # Generate basic maze & solve
    self.width = width
    self.height = height
    self.maze = Maze(width=width, height=height, generator=DepthFirstSearchGenerator())
    self.solver = MazeSolver()
    self.path = self.solver.solve(self.maze)
    self.maze.path = self.path    # No idea why this is necessary
    self.agent_cell = self.maze.start_cell
    self.num_actions = 4
    self.path_history = []  # (state, reward, done, info)

    # Add reward traces to cells
    self.reward_trace = self.calculate_reward_trace(trace_length)

  def calculate_reward_trace(self, trace_length):
    reward_trace = np.full((self.height, self.width), np.inf)
    goal = self.maze.end_cell.coordinates
    queue = deque([(goal, 0)])  # (cell coordinates, distance)
    visited = set()

    while queue:
      (x, y), dist = queue.popleft()
      if (x, y) in visited or dist > trace_length:
        continue
      visited.add((x, y))
      reward_trace[y, x] = dist

      for direction in Direction:
        if direction in self.maze[x, y].open_walls:
          neighbor = self.maze.neighbor(self.maze[x, y], direction)
          queue.append((neighbor.coordinates, dist + 1))

    # Normalize reward trace to go from large to small values
    max_dist = np.max(reward_trace[reward_trace != np.inf])
    reward_trace = max_dist - reward_trace
    reward_trace[reward_trace > trace_length] = 0  # Cut off at trace_length
    reward_trace[reward_trace == -np.inf] = 0  # Replace np.inf with 0
    return reward_trace.T

  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()

    # Box around maze
    ax.plot([-0.5, self.width-1+0.5], [-0.5, -0.5], color='black')
    ax.plot([-0.5, self.width-1+0.5], [self.height-1+0.5, self.height-1+0.5], color='black')
    ax.plot([-0.5, -0.5], [-0.5, self.height-1+0.5], color='black')
    ax.plot([self.width-1+0.5, self.width-1+0.5], [-0.5, self.height-1+0.5], color='black')

    # Plot maze
    for row in range(self.height):
      for column in range(self.width):
        # Path
        cell = self.maze[column, row]  # Tranpose maze coordinates (just how the maze is stored)
        if cell == self.maze.start_cell:
          ax.plot(row, column, 'go')
        elif cell == self.maze.end_cell:
          ax.plot(row, column,'bo')
        elif cell in self.maze.path:
          ax.plot(row, column, 'ro')

        # Walls
        if Direction.S not in cell.open_walls:
          ax.plot([row-0.5, row+0.5], [column+0.5, column+0.5], color='black')
        if Direction.E not in cell.open_walls:
          ax.plot([row+0.5, row+0.5], [column-0.5, column+0.5], color='black')

    return ax

  def reset(self):
    self.agent_cell = self.maze.start_cell
    self.path_history = []
    return self.agent_cell, {}

  # Takes action
  # Returns next state, reward, done, info
  def step(self, action):
    # Transform action into Direction
    if action == 0:
      action = Direction.N
    elif action == 1:
      action = Direction.E
    elif action == 2:
      action = Direction.S
    elif action == 3:
      action = Direction.W

    # Check if action runs into wall
    if action not in self.agent_cell.open_walls:
      self.path_history.append((self.agent_cell.coordinates, -0.01, False, {}))
      return self.agent_cell, -1, False, {}

    # Move agent
    else:
      prev_cell = self.agent_cell
      self.agent_cell = self.maze.neighbor(self.agent_cell, action)
      if self.agent_cell == self.maze.end_cell:    # Check if agent has reached the end
        self.path_history.append((self.agent_cell.coordinates, 1, True, {}))
        return self.agent_cell, 1, True, {}
      else:
        prev_trace = self.reward_trace[*prev_cell.coordinates]
        new_trace = self.reward_trace[*self.agent_cell.coordinates]
        reward = 1 if new_trace > prev_trace else 0
        self.path_history.append((self.agent_cell.coordinates, 0, False, {}))
        return self.agent_cell, reward, False, {}

  def save(self, filename):
    with open(filename, 'wb') as f:
      pkl.dump(self, f)

  def animate_history(self, run_history, file_name='maze.gif'):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 4)
    out_s_ax = fig.add_subplot(gs[0, 0])
    asso_s_ax = fig.add_subplot(gs[0, 1])
    maze_ax = fig.add_subplot(gs[0, 2:])
    # self.plot(maze_ax)
    def update(i):
      # Maze axis
      maze_ax.clear()
      self.plot(maze_ax)
      maze_ax.plot(self.path_history[i][0][1], self.path_history[i][0][0], 'yo')
      maze_ax.set_title(f'Step {i}, Reward: {self.path_history[i][1]}')
      maze_ax.set_aspect('equal')
      # Out spikes axis
      out_shape = run_history[i][-1].T.shape
      out_s_ax.clear()
      out_s_ax.imshow(run_history[i][-1].T)
      for y in range(0, out_shape[0], 20):    # MANUAL SET NUMBER
        out_s_ax.plot([0, out_shape[1]-1], [y, y], color='red')
      out_s_ax.set_title('Output spikes')
      out_s_ax.set_ylabel('Output neurons')
      out_s_ax.set_xlabel('Time')
      # Asso spikes axis
      asso_s_ax.clear()
      asso_s_ax.imshow(run_history[i][0].T)
      asso_s_ax.set_title('Asso. spikes')
      asso_s_ax.set_ylabel('Asso. neurons')
      asso_s_ax.set_xlabel('Time')
      fig.tight_layout()

    ani = FuncAnimation(plt.gcf(), update, frames=len(self.path_history), repeat=False)
    ani.save(file_name, writer='ffmpeg', fps=2)

class Grid_Cell_Maze_Environment(Maze_Environment):
  def __init__(self, width, height, trace_length=5, samples_file='Data/preprocessed_recalls_sorted.pkl'):
    super().__init__(width, height, trace_length)

    # Load spike train samples
    # {position: [spike_trains]}
    with open(samples_file, 'rb') as f:
      self.samples = pkl.load(f)

  # Returns:
  # - Spike train of grid cell corresponding to agent's position
  # - True coordinates (x, y)
  # - info: (empty)
  def reset(self):
    cell, info = super().reset()
    return self.state_to_grid_cell_spikes(cell), cell.coordinates, info

  def step(self, action):
    obs, reward, done, info = super().step(action)
    coords = obs.coordinates
    obs = self.state_to_grid_cell_spikes(obs)
    return obs, reward, done, coords, info

  def state_to_grid_cell_spikes(self, cell):
    return random.choice(self.samples[cell.coordinates])


if __name__ == '__main__':
  from train_DQN import DQN, ReplayMemory
  from scripts.Chris.DQN.train_DQN import run_episode

  device = 'cpu'
  n_actions = 4
  input_size = 300
  lr = 0.01
  policy_net = DQN(input_size, n_actions).to(device)
  target_net = DQN(input_size, n_actions).to(device)
  target_net.load_state_dict(policy_net.state_dict())
  optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
  memory = ReplayMemory(1000)
  env = Grid_Cell_Maze_Environment(width=5, height=5)

  run_episode(env, policy_net, 'cpu', 100, eps=0.9)
  env.animate_history()

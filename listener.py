import math
import subprocess
import os
import random
import matplotlib.pyplot as plt
import matplotlib
from itertools import count
import pyautogui
import time
from pynput.keyboard import Key, Controller
from threading import Thread

keyboard = Controller()

import torch
import torch.optim as optim

from replayMemory import Transition, ReplayMemory
from dqn import DQN
from training_loop import optimize_model

# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
#
# plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

os.chdir('pong_master')

process = None


def open_game():
    global process
    process = subprocess.Popen(
        ['python', 'game.py'],
        stdout=subprocess.PIPE,
        text=True)

    while True:
        line = process.stdout.readline().strip().split()
        print(line)
        if line[0] == "pygame":
            time.sleep(1)
            pyautogui.press("up")
            pyautogui.press("up")
            time.sleep(0.5)
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
            time.sleep(1)
            pyautogui.press("up")
            pyautogui.press("up")
            time.sleep(0.5)
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
            break


# open_game()


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.95
LR = 1e-4

# Get number of actions
n_actions = 2
# Get the number of state observations
n_observations = 3

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 1)]], device=device, dtype=torch.long)


# episode_durations = []
#
#
# def plot_durations(show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

open_game()
time.sleep(0.1)

for i_episode in range(num_episodes):
    # Initialize the environment and get its state

    line = process.stdout.readline().strip().split()
    try:
        line = [int(x) for x in line]
        state = torch.tensor(line[0:3], device=device)
        state = torch.tensor(
            [(x - torch.min(state)) / (torch.max(state - torch.min(state))) * 2 - 1 for x in state], device=device)
        if not line:
            break

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            if action == 1:
                pyautogui.keyUp("up")
                pyautogui.keyDown("up")
            elif action == 0:
                pyautogui.keyUp("down")
                pyautogui.keyDown("down")
            # observation, reward, terminated, truncated, _ = env.step(action.item())

            line = process.stdout.readline().strip().split()
            line = [int(x) for x in line]
            observation = torch.tensor(line[0:3], device=device)
            observation = torch.tensor(
                [(x - torch.min(observation)) / (torch.max(observation - torch.min(observation))) * 2 - 1 for x in
                 observation], device=device)

            print(line)

            reward = line[4] - line[3]

            reward = torch.tensor([reward], device=device)
            done = line[3] > 0

            print(reward)

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(device=device,
                           memory=memory,
                           optimizer=optimizer,
                           policy_net=policy_net,
                           target_net=target_net,
                           BATCH_SIZE=BATCH_SIZE,
                           GAMMA=GAMMA)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                        1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                process.terminate()
                open_game()
                break
    except ValueError:
        pass


# print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()

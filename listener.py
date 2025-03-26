import redis
import subprocess
import os
import random
from replayMemory import Transition, ReplayMemory

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


os.chdir('pong-master')

process = subprocess.Popen(
    ['python', 'game.py'],
    stdout=subprocess.PIPE,
    text=True
)

while True:
    line = process.stdout.readline().strip().split()
    try:
        line = [int(x) for x in line]
        state = torch.FloatTensor(line[0:3])
        state = [(x - torch.min(state)) / (torch.max(state - torch.min(state))) for x in state]
        if not line:
            break
        print(f"Odebrano: {line}")
    except ValueError:
        pass
        



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""Création de l'architecture du réseau neuronal"""

class ReseauNeuronal(nn.Module):
    def __init__(self, taille_etat, taille_action, seed=42):
        super(ReseauNeuronal, self).__init__()
        self.graine = torch.manual_seed(seed)
        self.couche_connectee_1 = nn.Linear(taille_etat, 64)
        self.couche_connectee_2 = nn.Linear(64, 64)
        self.couche_connectee_3 = nn.Linear(64, taille_action)

    def forward(self, etat):
        x = self.couche_connectee_1(etat)
        x = F.relu(x)
        x = self.couche_connectee_2(x)
        x = F.relu(x)
        return self.couche_connectee_3(x)

import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
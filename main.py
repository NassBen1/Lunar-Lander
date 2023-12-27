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

#Mise en place de l'environnement et récupération des informations sur l'état et les actions

import gym
env = gym.make("LunarLander-v2")
forme_etat = env.observation_space.shape
print("Forme de l'état =", forme_etat)
taille_etat = env.observation_space.shape[0]
print("Taille de l'état =", taille_etat)
nb_actions = env.action_space.n
print("Nombre d'actions =", nb_actions)

#Initialisation des hyperparamètres

taux_apprentissage = 5e-4
taille_mini_batch = 100
facteur_discount = 0.98
taille_memoire_replay = int(1e5)
parametre_interpolation = 1e-3

#Mise en place du replay memory

class MemoireReplay(object):



    """Initialisation du replay memory"""
    def __init__(self, capacite):
        self.appareil = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacite = capacite
        self.memoire = []

    """Ajout d'un événement dans le replay memory"""
    def push(self, evenement):
        self.memoire.append(evenement)
        if len(self.memoire) > self.capacite:
            del self.memoire[0]

    def echantillon(self, taille_batch):
        experiences = random.sample(self.memoire, k=taille_batch)
        etats = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.appareil)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.appareil)
        recompenses = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.appareil)
        etats_suivants = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.appareil)
        termines = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.appareil)
        return etats, etats_suivants, actions, recompenses, termines

"""Implémentation de la classe DQN"""

class Agent():

    def __init__(self, taille_etat, taille_action):
        self.appareil = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.taille_etat = taille_etat
        self.taille_action = taille_action
        self.reseau_local_Q = ReseauNeuronal(taille_etat, taille_action).to(self.appareil)
        self.reseau_cible_Q = ReseauNeuronal(taille_etat, taille_action).to(self.appareil)
        self.optimiseur = optim.Adam(self.reseau_local_Q.parameters(), lr=taux_apprentissage)
        self.memoire = MemoireReplay(taille_memoire_replay)
        self.etapes_temps = 0


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F

from collections import deque

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


# Mise en place de l'environnement et récupération des informations sur l'état et les actions

import gym

env = gym.make("LunarLander-v2")
forme_etat = env.observation_space.shape
print("Forme de l'état =", forme_etat)
taille_etat = env.observation_space.shape[0]
print("Taille de l'état =", taille_etat)
nb_actions = env.action_space.n
print("Nombre d'actions =", nb_actions)

# Initialisation des hyperparamètres

taux_apprentissage = 5e-4
taille_mini_batch = 100
facteur_discount = 0.98
taille_memoire_replay = int(1e5)
parametre_interpolation = 1e-3


# Mise en place du replay memory

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
        recompenses = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(
            self.appareil)
        etats_suivants = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(
            self.appareil)
        termines = torch.from_numpy(
            np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.appareil)
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

    def etape(self, etat, action, recompense, etat_suivant, termine):
        self.memoire.push((etat, action, recompense, etat_suivant, termine))
        self.etapes_temps = (self.etapes_temps + 1) % 4
        if self.etapes_temps == 0:
            if len(self.memoire.memoire) > taille_mini_batch:
                experiences = self.memoire.echantillon(100)
                self.apprendre(experiences, facteur_discount)

    def agir(self, etat, epsilon=0.):
        etat = torch.from_numpy(etat).float().unsqueeze(0).to(self.appareil)
        self.reseau_local_Q.eval()
        with torch.no_grad():
            valeurs_action = self.reseau_local_Q(etat)
        self.reseau_local_Q.train()

        if random.random() > epsilon:
            return np.argmax(valeurs_action.cpu().data.numpy())
        else:
            return random.randint(0, self.taille_action - 1)

    def apprendre(self, experiences, facteur_discount):
        etats, etats_suivants, actions, recompenses, termines = experiences
        prochaines_valeurs_Q_cibles = self.reseau_cible_Q(etats_suivants).detach().max(1)[0].unsqueeze(1)
        valeurs_Q_cibles = recompenses + (facteur_discount * prochaines_valeurs_Q_cibles * (1 - termines))
        valeurs_Q_attendues = self.reseau_local_Q(etats).gather(1, actions)
        perte = F.mse_loss(valeurs_Q_attendues, valeurs_Q_cibles)
        self.optimiseur.zero_grad()
        perte.backward()
        self.optimiseur.step()
        self.mise_a_jour_progressive(self.reseau_local_Q, self.reseau_cible_Q, parametre_interpolation)

    def mise_a_jour_progressive(self, modele_local, modele_cible, parametre_interpolation):
        for parametre_cible, parametre_local in zip(modele_cible.parameters(), modele_local.parameters()):
            parametre_cible.data.copy_(parametre_interpolation * parametre_local.data + (
                    1.0 - parametre_interpolation) * parametre_cible.data)

    """Initialisation de l'Agent DQN"""


agent = Agent(taille_etat, nb_actions)

num_episodes = 500
max_num_time_steps = 1000
epsilon_debut = 1.0
epsilon_fin = 0.01
taux_de_decay_epsilon = 0.995
epsilon = epsilon_debut
scores_sur_100_episodes = deque(maxlen=100)

for episode in range(1, num_episodes + 1):
    etat, _ = env.reset()
    score = 0
    for pas_temporel in range(max_num_time_steps):
        action = agent.agir(etat, epsilon)
        etat_suivant, recompense, termine, _, _ = env.step(action)
        agent.etape(etat, action, recompense, etat_suivant, termine)
        etat = etat_suivant
        score += recompense
        if termine:
            break
    scores_sur_100_episodes.append(score)
    epsilon = max(epsilon_fin, taux_de_decay_epsilon * epsilon)
    print('\rEpisode {}\tScore Moyen: {:.2f}'.format(episode, np.mean(scores_sur_100_episodes)), end="")
    if episode % 100 == 0:
        print('\rEpisode {}\tScore Moyen: {:.2f}'.format(episode, np.mean(scores_sur_100_episodes)))
    if np.mean(scores_sur_100_episodes) >= 200.0:
        print('\rEnvironnement résolu en {:d}\tScore Moyen: {:.2f}'.format(episode - 100,
                                                                           np.mean(scores_sur_100_episodes)))
        torch.save(agent.reseau_local_Q.state_dict(), 'checkpoint.pth')
        break

"""Visualizing the results"""

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import gym
import imageio
import glob
import io
import base64
from IPython.display import HTML

def afficher_video_du_modele(agent, nom_env):
    env = gym.make(nom_env, render_mode='rgb_array')
    etat, _ = env.reset()
    termine = False
    images = []
    while not termine:
        image = env.render()
        images.append(image)
        action = agent.agir(etat)
        etat, recompense, termine, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', images, fps=30)

afficher_video_du_modele(agent, 'LunarLander-v2')

def afficher_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encode = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encode.decode('ascii'))))
    else:
        print("Impossible de trouver la vidéo")

afficher_video()
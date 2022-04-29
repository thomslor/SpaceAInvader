import copy
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

from game.SpaceInvaders import SpaceInvaders
from controller.epsilon_profile import EpsilonProfile


class DQNAgent():
    """ 
    Cette classe d'agent représente un agent utilisant l'algorithme DQN pour mettre 
    à jour sa politique d'action.
    """
    TEST_FREQUENCY = 100

    def __init__(self, model: nn.Module, eps_profile: EpsilonProfile, gamma: float, alpha: float, replay_memory_size: int = 1000, batch_size: int = 32, target_update_freq: int = 100, tau: float = 1., final_exploration_episode : int = 500):
        """
        Ce constructeur initialise une nouvelle instance de la classe QAgent.
        Il doit stocker les différents paramètres nécessaires au fonctionnement 
        de l'algorithme et initialiser la fonction de valeur d'action, notée Q.

        :param maze: Le labyrinthe à résoudre 
        :type maze: Maze
        :param eps_profile: Le profil du paramètre d'exploration epsilon 
        :type eps_profile: EpsilonProfile
        :param gamma: Le facteur d'atténuation
        :type gamma: float
        :param alpha: Le taux d'apprentissage
        :type alpha: float
        :param tau: Le taux de mise à jour du réseau cible
        :type tau: float
        :param target_update_frequency: La fréquence de mise à jour du réseau cible
        :type target_update_frequency: int
        :param batch_size: La taille de l'échantillon utilisé pour la mise à jour du 
        réseau Q
        :type batch_size: int
        """
        print(model)
        self.policy_net = model
        self.target_net = copy.deepcopy(model)

        # Paramètres d'apprentissage
        self.alpha = alpha
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = batch_size
        self.target_update_frequency = target_update_freq
        self.tau = tau

        # Profil du Epsilon
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial
        self.init_epsilon = self.eps_profile.initial
        self.final_epsilon = self.eps_profile.final
        self.final_exploration_episode = final_exploration_episode

        # Cirtère d'optimisation
        self.criterion = nn.MSELoss()

        # Méthode de descente de gradient
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)

        self.na = 4

    def init_replay_memory(self, env: SpaceInvaders):
        """Cette méthode initialise le buffer d'expérience replay.

        :param env: Environnement (le jeu)
        :type env: SpaceInvaders
        """
        # Replay memory pour s, a, r, terminal, and sn
        self.Ds = np.zeros([self.replay_memory_size, 2], dtype=np.float32)
        self.Da = np.zeros([self.replay_memory_size, env.na], dtype=np.float32)
        self.Dr = np.zeros([self.replay_memory_size], dtype=np.float32)
        self.Dt = np.zeros([self.replay_memory_size], dtype=np.float32)
        self.Dsn = np.zeros([self.replay_memory_size, 2], dtype=np.float32)

        self.d = 0     # counter for storing in D
        self.ds = 0    # total number of steps
        
    def updateQ(self, state, action, reward, next_state, terminal):
        """ Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).

        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        # Ajoute les éléments dans le buffer d'expérience
        
        
        self.Ds[self.d], self.Dr[self.d], self.Dsn[self.d], self.Dt[self.d] = state, reward, next_state, terminal

        # since Da[d,:] is a one-hot vector
        self.Da[self.d] = 0
        self.Da[self.d, action] = 1

        # since D is a circular buffer
        self.d = (self.d + 1) % self.replay_memory_size
        self.ds = self.ds + 1

        # Commence l'apprentissage quand le buffer est plein
        if self.ds >= self.replay_memory_size:

            self.optimizer.zero_grad()

            # Sélectionne des indices aléatoires dans le buffer
            c = np.random.choice(self.replay_memory_size, self.minibatch_size)
            

            # Récupère les batch de données associés
            x_batch, a_batch, r_batch, y_batch, t_batch = torch.from_numpy(self.Ds[c]), torch.from_numpy(
                self.Da[c]), torch.from_numpy(self.Dr[c]),  torch.from_numpy(self.Dsn[c]), torch.from_numpy(self.Dt[c])

            # Calcul de la valeur courante 
            current_value = self.policy_net(x_batch).gather(1, a_batch.max(1).indices.unsqueeze(1)).squeeze(1)

            # Calcul de la valeur cible
            target_value = self.target_net(y_batch).max(1).values * self.gamma * (1. - t_batch) + r_batch

            # La fonction 'detach' arrête la rétropopagation du gradient à 
            # travers la partie du graphe concernée (ici target network)
            loss = self.criterion(current_value, target_value.detach())

            loss.backward()
            self.optimizer.step()

    def learn(self, env, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.

        :param env: L'environnement 
        :type env: gym.Env
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int
        """
        self.na = 4
        self.init_replay_memory(env)

        # Initialisation des stats d'apprentissage
        sum_rewards = np.zeros(n_episodes)
        len_episode = np.zeros(n_episodes)
        n_steps = np.zeros(n_episodes) + max_steps

        start_time = time.time()

        # Execute N episodes
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()
            print("Episode : ", episode)
            
            # Execute K steps
            for step in range(max_steps):
                
                # Selectionne une action
                action = self.select_action(state)

                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)
                
                # Stocke les données d'apprentissage
                sum_rewards[episode] += reward
                

                print("Rewards : ", sum_rewards[episode])
                len_episode[episode] += 1

                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state, terminal)

                if reward == 1:
                    break

                if terminal:
                    n_steps[episode] = step + 1  # number of steps taken
                    break

                state = next_state

                

            self.epsilon = max(self.final_epsilon, self.epsilon - (1. / self.final_exploration_episode))
            # self.epsilon = max(self.epsilon - self.eps_profile.dec_step, self.eps_profile.final)

            # Mets à jour le réseau cible, en copiant tous les weights et biases dans DQN
            if n_episodes % self.target_update_frequency == 0:
                if (self.tau < 1.):
                    # Mets à jour le réseau de neurones cible en lissant ses paramètres avec ceux de policy_net
                    self.soft_update(self.tau)
                else:
                    # Copie le réseau de neurones courant dans le réseau cible
                    self.hard_update()

            n_ckpt = 10
           
    

    def select_action(self, state : 'Tuple[int, int]'):
        """À COMPLÉTER!
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).

        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na)      # random action
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state):
        """
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """
        return self.policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax()

    def hard_update(self):
        """ Cette fonction copie le réseau de neurones 
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self, tau):
        """ Cette fonction fait mise à jour glissante du réseau de neurones cible 
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau) * target_param.data)

    def run_tests(self, env, n_runs, max_steps):
        test_score = 0.
        for k in range(n_runs):
            s = env.reset()
            for t in range(max_steps):
                q = self.policy_net(torch.FloatTensor(s).unsqueeze(0))
                # greedy action with random tie break
                a = np.random.choice(np.where(q[0] == q[0].max())[0])
                sn, r, terminal = env.step(a)
                test_score += r
                if terminal:
                    break
                s = sn
        return test_score / n_runs

from os import stat
import numpy as np
from controller import AgentInterface
from game.SpaceInvaders import SpaceInvaders
from controller.epsilon_profile import EpsilonProfile
import pandas as pd
import matplotlib.pyplot as plt

class QAgent(AgentInterface):
    """ 
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning 
    pour mettre à jour sa politique d'action.
    """

    def __init__(self, game: SpaceInvaders, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        """A LIRE
        Ce constructeur initialise une nouvelle instance de la classe QAgent.
        Il doit stocker les différents paramètres nécessaires au fonctionnement de l'algorithme et initialiser la 
        fonction de valeur d'action, notée Q.

        :param maze: Le labyrinthe à résoudre 
        :type maze: Maze

        :param eps_profile: Le profil du paramètre d'exploration epsilon 
        :type eps_profile: EpsilonProfile
        
        :param gamma: Le discount factor 
        :type gamma: float
        
        :param alpha: Le learning rate 
        :type alpha: float

        - Visualisation des données
        :attribut mazeValues: la fonction de valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        :penser à bien stocker aussi la taille du labyrinthe (nx,ny)

        :attribut qvalues: la Q-valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        """
        
        # Initialise la fonction de valeur Q
        # Implémentation Distance
        # self.Q = np.zeros([11,2,5,2,4])
        
        # Implémentation position
        self.Q = np.zeros([10,10,5,2,4])
        

        self.game = SpaceInvaders
        self.na = 4

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        

    def learn(self, env, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.

        :param env: L'environnement 
        :type env: gym.Envselect_action
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int

        # Visualisation des données
        Elle doit proposer l'option de stockage de (i) la fonction de valeur & (ii) la Q-valeur 
        dans un fichier de log
        """
        n_steps = np.zeros(n_episodes) + max_steps
        
        # Execute N episodes 
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()
            # Execute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)

                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)
                
                if terminal:
                    n_steps[episode] = step + 1  
                    break

                if reward == 1:
                    n_steps[episode] = step + 1
                    break

                state = next_state
                
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)

            # Sauvegarde et affiche les données d'apprentissage
            if n_episodes >= 0:
                state = env.reset()
                state = tuple(map(int, state))
                print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[state][self.select_greedy_action(state)]), end =" ")
                ## self.save_log(env, episode)
        
        # Visualisation de l'apprentissage
        p_step = [1-int(n)/max_steps for n in n_steps]
        p_step_cumul = np.cumsum(p_step)
        plt.plot(p_step_cumul)
        plt.xlabel('Episode')
        plt.ylabel('% of steps')
        plt.title('Learning curve')
        plt.legend()
        plt.show()



    def updateQ(self, state, action, reward, next_state):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).

        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        state = tuple(map(int, state))
        next_state = tuple(map(int, next_state))
        print(state, " : ", next_state)
        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))

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

    def select_greedy_action(self, state : 'Tuple[int, int]'):
        """
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """
        state = tuple(map(int, state))
        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])
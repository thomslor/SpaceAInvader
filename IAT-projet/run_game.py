from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

from controller.epsilon_profile import EpsilonProfile
from controller.networks import MLP
from controller.dqn_agent import DQNAgent
from controller.qagent import QAgent

from matplotlib import pyplot as plt



# Hyperparamètres 
n_episodes = 200
max_steps = 2000
alpha = 0.1
gamma = 1.
eps_profile = EpsilonProfile(0.7, 0.15)

# Passer à True pour voir le comportement de l'agent
game = SpaceInvaders(display=False)

# model = MLP()
# controller = KeyboardController()
# controller = DQNAgent(model, eps_profile, gamma, alpha)
# controller = RandomAgent(game.na)
controller = QAgent(game, eps_profile, gamma, alpha)
controller.learn(game, n_episodes, max_steps)
print("Phase d'entraînement terminée")

print("Phase de test")
game.display = True
state = game.reset()

nb_tires = 0
tab = [[],[]]

sum_reward = 0

while nb_tires < 300:
    action = controller.select_action(state)
    state, reward, is_done = game.step(action)
    if is_done:
        print("Game over")
        break
    sum_reward += reward
    print(game.nb_tire, "  ", game.score_val)
    if (nb_tires != game.nb_tire and game.nb_tire > 0 and game.bullet_state == "rest"):
        tab[0].append(game.nb_tire)
        tab[1].append(game.score_val / game.nb_tire)
        nb_tires = game.nb_tire
    sleep(0.0001)
    
    

plt.plot(tab[0], tab[1])
plt.show()
exit()



from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

from controller.epsilon_profile import EpsilonProfile
from controller.networks import MLP
from controller.dqn_agent import DQNAgent

from matplotlib import pyplot as plt

def main():

     # Hyperparamètres basiques
    n_episodes = 1000
    max_steps = 1000
    alpha = 0.001
    gamma = 1.
    eps_profile = EpsilonProfile(1.0, 0.1)
    final_exploration_episode = 1000

    # Hyperparamètres de DQN
    eps_profile = EpsilonProfile(1.0, 0.1)
    batch_size = 32
    replay_memory_size = 1000
    target_update_frequency = 100
    tau = 1.0

    game = SpaceInvaders(display=True)
    model = MLP()
    # controller = KeyboardController()
    controller = DQNAgent(model, eps_profile, gamma, alpha)
    # controller = RandomAgent(game.na)
    controller.learn(game, n_episodes, max_steps)
    print("A appris fort!")
 
    state = game.reset()
    sum_reward=0

    nb_tires = 0
    tab = [[],[]]
    
    while nb_tires < 300:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sum_reward += reward
        print(game.nb_tire, "  ", game.score_val)
        if (nb_tires != game.nb_tire and game.nb_tire > 0 and game.bullet_state == "rest"):
            tab[0].append(game.nb_tire)
            tab[1].append(game.score_val / game.nb_tire)
            nb_tires = game.nb_tire
        sleep(0.0001)
        
        
    
    plt.plot(tab[0], tab[1])
    plt.show()
    

if __name__ == '__main__' :
    main()

import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        """
        Ce constructeur crée une instance de réseau de neurones de type Multi Layer Perceptron (MLP).
        L'architecture choisie doit être choisie de façon à capter toute la complexité du problème
        sans pour autant devenir intraitable (trop de paramètres d'apprentissages). 
        """
    

        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        """Cette fonction réalise un passage dans le réseau de neurones. 

        :param x: L'état
        :return: Le vecteur de valeurs d'actions (une valeur par action)
        """
        x = self.flatten(x)
        qvalues = self.layers(x)
        return qvalues

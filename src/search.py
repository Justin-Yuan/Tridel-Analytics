from __future__ import division, print_function, absolute_import 

import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt

from AE import Autoencoder


if __name__ == '__main__':
    # perform grid search to find the bset hyperparameters 

    score = 99999
    num_keep = 10
    num_model_eval = 3
    top_models_score_based = []
    top_models_loss_based = []
    top_models_comb_based = []   # sum of score and loss 

    # size, bedroom, bathroom, neighborhood 

    # total trials: 4*6*8*15 = 2880 different settings 
    for i in range(0, 2, 0.5):  # search space for size 
        for j in range(0, 3, 0.5):  # search space for bedroom 
            for k in range(20, 100, 10):  # search space for bathroom 
                for p in range(5, 20, 1):  # search space for neighborhood 
                    for q in range(num_model_eval):
                        # train a model with the current set of penalty weights 
                        ae = Autoencoder()
                        ae.build()
                        ae.train()
                        score = ae.evaluate()

                        # save top 10 models with best performance 

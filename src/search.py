from __future__ import division, print_function, absolute_import 

import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt

from AE import *


if __name__ == '__main__':
    # perform grid search to find the bset hyperparameters 

    

    score = 99999
    neigh_penalty = 0
    num_keep = 10
    num_model_eval = 3
    top_models_score_based = []
    top_models_loss_based = []
    top_models_comb_based = []   # sum of score and loss 

    # size, bedroom, bathroom, neighborhood 

    for p in range(5, 20, 1):  # search space for neighborhood 
        for q in range(num_model_eval):
            # train a model with the current set of penalty weights 
            current = test_AE(neigh_penalty=p)
            if current > score:
                score = current 
                neigh_penalty = p
                        

            # save top 10 models with best performance 

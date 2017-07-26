from __future__ import division, print_function, absolute_import 

import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt

from AE import Autoencoder


if __name__ == '__main__':
    # perform grid search to find the bset hyperparameters 

    score = 99999

    for i in range():
        for j in range():
            for k in range():
                ae = Autoencoder()
                ae.build()
                ae.train()
                score = ae.evaluate()
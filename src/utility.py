from __future__ import division, print_function, absolute_import 

import numpy as np
import pandas as pd 


def get_data(path):
    df = pd.read_csv(path)
    matrix_df = df.loc[:, 'Bedrooms':'housing price'].as_matrix()[:3538]
    mat = matrix_df[:,:]
    where_are_NaNs = np.isnan(mat)
    mat[where_are_NaNs] = 0
    maximums = np.amax(mat, axis=0)
    mat = mat/maximums
    return mat 


def next_batch(mat, index, size):
    if index + size <= mat.shape[0]:
        return index+size, mat[index:index+size]
    else:
        return index+size-mat.shape[0], np.concatenate((mat[index:], mat[:index+size-mat.shape[0]]), axis=0)


def evaluate_by_std(predicted_scores, comp_sequence):
    """ 
    variance of differences or even powers 
    """
    differences = predicted_scores - comp_sequence
    return np.std(differences)


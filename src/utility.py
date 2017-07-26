from __future__ import division, print_function, absolute_import 

import numpy as np
import pandas as pd 


def get_data(path):
    df = pd.read_csv(path)
    matrix_df = df.as_matrix()
    mat = matrix_df[:,:]
    where_are_NaNs = np.isnan(mat)
    mat[where_are_NaNs] = 0
    maximums = np.amax(mat, axis=0)
    print(maximums)
    mat = mat/maximums
    return mat 


def next_batch(mat, index, size):
    if index + size <= mat.shape[0]:
        return index+size, mat[index:index+size]
    else:
        return index+size-mat.shape[0], np.concatenate((mat[index:], mat[:index+size-mat.shape[0]]), axis=0)



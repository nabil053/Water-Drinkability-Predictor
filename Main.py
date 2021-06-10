import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

df = pd.read_csv('water_potability.csv')
df = df.dropna().reset_index(drop=True)
#print(df)

size_training = (3 * df.shape[0])//5
size_validation = df.shape[0]//5
size_testing = df.shape[0] - (size_training + size_validation)
#print(size_training, size_validation, size_testing)

df_train = (df.iloc[0:size_training,:]).reset_index(drop=True)
df_cv = (df.iloc[size_training:(size_training + size_validation),:]).reset_index(drop=True)
df_test = (df.iloc[(size_training + size_validation):,:]).reset_index(drop=True)
#print(df_train.columns)

w = np.zeros(df_train.shape[1])
matrix_train = np.ones((df_train.shape[0], df_train.shape[1]))
matrix_train[:,1:] = df_train.iloc[:,:-1]
alpha = 0.001
eta = alpha

temp = np.apply_along_axis(lambda x : np.matmul(x,w), 1, matrix_train)
temp = 1 / (1 + np.exp(temp*(-1)))
temp = df_train.iloc[:,-1:].to_numpy().flatten() - temp
temp = np.apply_along_axis(lambda x : np.multiply(x, temp), 0, matrix_train)

partial = np.apply_along_axis(lambda x : np.sum(x), 0, temp)
partial_sqrt = np.linalg.norm(partial)

w = w + (eta * partial)
#print(w)


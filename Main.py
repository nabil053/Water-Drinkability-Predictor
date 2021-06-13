import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

#Reading data from dataset and dropping data points with at least one null value
df = pd.read_csv('water_potability.csv')
df = df.dropna().reset_index(drop=True)
#print(df)

#Determining the sizes of training, validation and test sets
size_training = (3 * df.shape[0])//5
size_validation = df.shape[0]//5
size_testing = df.shape[0] - (size_training + size_validation)
#print(size_training, size_validation, size_testing)

#Splitting the dataframe into training, validation and test sets
df_train = (df.iloc[0:size_training,:]).reset_index(drop=True)
df_cv = (df.iloc[size_training:(size_training + size_validation),:]).reset_index(drop=True)
df_test = (df.iloc[(size_training + size_validation):,:]).reset_index(drop=True)
#print(df_train.columns)

#Initializing parameters vector to 0 and copying features of training data to a matrix with zero feature included
w = np.zeros(df_train.shape[1])
matrix_train = np.ones((df_train.shape[0], df_train.shape[1]))
matrix_train[:,1:] = df_train.iloc[:,:-1]

#Normalizing the data in the matrix
for i in range(1,matrix_train.shape[1]):
    mean = np.mean(matrix_train[:,i])
    std = np.std(matrix_train[:,i])
    matrix_train[:,i] = (matrix_train[:,i] - mean)/std
#print(matrix_train)

#Initializing step size and stopping criteria
alpha = 0.01
stepsize = alpha
stop_criteria = 0.001
#print(matrix_train)

#Iteration for determining parameters using gradient ascent
i = 1
while(True):
    #Intermediate calculations of determining partial derivative of maximum likelihood
    temp = np.apply_along_axis(lambda x: np.matmul(x, w), 1, matrix_train)
    for j in range(len(temp)):
        if temp[j] > 10:
            temp[j] = 1
        elif temp[j] < -10:
            temp[j] = 0
        else:
            temp[j] = 1 / (1 + math.exp(temp[j] * (-1)))
    temp = df_train.iloc[:, -1:].to_numpy().flatten() - temp
    temp = np.apply_along_axis(lambda x: np.multiply(x, temp), 0, matrix_train)

    #Final result of maximum likelihood's partial derivative vector, as well as its magnitude
    partial = np.apply_along_axis(lambda x: np.sum(x), 0, temp)
    partial_sqrt = np.linalg.norm(partial)

    #Updating the parameters vector
    w = w + (stepsize * partial)

    #Updating the stepsize
    stepsize = alpha / (i + 1)

    #If not the first step and the change in magnitude of partial derivative is below criteria value, end iteration
    if i != 1 and math.fabs(partial_sqrt - partial_sqrt_prev) <= stop_criteria:
        break

    partial_sqrt_prev = partial_sqrt
    i = i + 1

print(i, w)
true_zero = 0
false_zero = 0
true_one = 0
false_one = 0
is_zero = False
is_one = False
for i in range(df_train.shape[0]):
    #d = df_train.iloc[i,:-1].to_numpy()
    d = matrix_train[i,:]
    #d = np.insert(d, 0, 1)
    t = np.matmul(d,w)
    if t > 10:
        is_one = True
        is_zero = False
    elif t < -10:
        is_one = False
        is_zero = True
    else:
        tp = 1 / (1 + math.exp((-1) * t))
        if tp > 0.5:
            is_one = True
            is_zero = False
        else:
            is_one = False
            is_zero = True
    if (df_train.iloc[i,-1] == 0) and is_zero:
        true_zero = true_zero + 1
    elif (df_train.iloc[i,-1] == 1) and is_zero:
        false_zero = false_zero + 1
    elif (df_train.iloc[i,-1] == 1) and is_one:
        true_one = true_one + 1
    elif (df_train.iloc[i,-1] == 0) and is_one:
        false_one = false_one + 1
#print(math.exp(np.matmul(d,w)))
# print(true_zero)
# print(false_zero)
# print(true_one)
# print(false_one)

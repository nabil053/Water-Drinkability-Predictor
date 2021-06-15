import math
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

#Reading data from dataset and dropping data points with at least one null value, because null data with impact the result
df = pd.read_csv('water_potability.csv')
df = df.dropna().reset_index(drop=True)

#Modifying the dataframe to make the number of data of both classes the same, since class 0 has overwhelmingly more data
df_ones = df.loc[df['Potability'] == 1]
df_zeros = df.loc[df['Potability'] == 0]
df_zeros = df.iloc[0:df_ones.shape[0],:]
df_ones = pd.concat([df_ones, df_zeros])
df = df_ones.reset_index(drop=True)

#Determining the sizes of training, validation and test sets
size_training = (3 * df.shape[0])//5
size_validation = df.shape[0]//5
size_testing = df.shape[0] - (size_training + size_validation)

#Splitting the dataframe into training, validation and test sets
df_train = (df.iloc[0:size_training,:]).reset_index(drop=True)
df_cv = (df.iloc[size_training:(size_training + size_validation),:]).reset_index(drop=True)
df_test = (df.iloc[(size_training + size_validation):,:]).reset_index(drop=True)

#Copying features of training, cross validation and testing data to respective matrices with zero feature included
matrix_train = np.ones((df_train.shape[0], df_train.shape[1]))
matrix_train[:,1:] = df_train.iloc[:,:-1]
matrix_cv = np.ones((df_cv.shape[0], df_cv.shape[1]))
matrix_cv[:,1:] = df_cv.iloc[:,:-1]
matrix_test = np.ones((df_test.shape[0], df_test.shape[1]))
matrix_test[:,1:] = df_test.iloc[:,:-1]

#Normalizing the data in the matrices
for i in range(1,matrix_train.shape[1]):
    mean = np.mean(matrix_train[:,i])
    std = np.std(matrix_train[:,i])
    matrix_train[:,i] = (matrix_train[:,i] - mean)/std
for i in range(1,matrix_cv.shape[1]):
    mean = np.mean(matrix_cv[:,i])
    std = np.std(matrix_cv[:,i])
    matrix_cv[:,i] = (matrix_cv[:,i] - mean)/std
for i in range(1,matrix_test.shape[1]):
    mean = np.mean(matrix_test[:,i])
    std = np.std(matrix_test[:,i])
    matrix_test[:,i] = (matrix_test[:,i] - mean)/std

#Initializing parameter vector to 0 as well as starting step size and stopping criteria
alpha = 0.01
stop_criteria = 0.001
w = np.zeros(df_train.shape[1])

#Selecting some possible tuning parameter values and the variables needed to find the model with the best one
param_vals = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
min_error = df_cv.shape[0]
error = 0

#Carrying out gradient ascent for each possible value of tuning parameter
for p in param_vals:
    i = 1
    stepsize = alpha
    w_temp = np.zeros(df_train.shape[1])

    # Gradient ascent iteration for determining parameters using gradient ascent
    while(True):
        #Intermediate calculations of determining partial derivative of maximum likelihood
        temp = np.apply_along_axis(lambda x: np.matmul(x, w_temp), 1, matrix_train)
        for j in range(len(temp)):
            if temp[j] > 10:
                temp[j] = 1
            elif temp[j] < -10:
                temp[j] = 0
            else:
                temp[j] = 1 / (1 + math.exp(temp[j] * (-1)))
                if temp[j] > 0.5:
                    temp[j] = 1
                else:
                    temp[j] = 0
        temp = df_train.iloc[:, -1:].to_numpy().flatten() - temp
        temp = np.apply_along_axis(lambda x: np.multiply(x, temp), 0, matrix_train)

        #Final result of maximum likelihood's partial derivative vector, as well as its magnitude
        partial = np.apply_along_axis(lambda x: np.sum(x), 0, temp)
        partial_sqrt = np.linalg.norm(partial)

        #Updating the parameters vector
        w_temp = ((1 - (2 * p * stepsize)) * w_temp) + (stepsize * partial)

        #Updating the stepsize
        stepsize = alpha / (i + 1)

        #If not the first step and the change in magnitude of partial derivative is below criteria value, end iteration
        if i != 1 and math.fabs(partial_sqrt - partial_sqrt_prev) <= stop_criteria:
            break

        partial_sqrt_prev = partial_sqrt
        i = i + 1

    #Checking the error of current model's performance in cross validation data
    temp = np.apply_along_axis(lambda x: np.matmul(x, w_temp), 1, matrix_cv)
    for k in range(len(temp)):
        if temp[k] > 10:
            temp[k] = 1
        elif temp[k] < -10:
            temp[k] = 0
        else:
            temp[k] = 1 / (1 + math.exp(temp[k] * (-1)))
            if temp[k] > 0.5:
                temp[k] = 1
            else:
                temp[k] = 0
    temp = df_cv.iloc[:, -1:].to_numpy().flatten() - temp
    error = (np.absolute(temp)).sum()

    #Updating the tuning parameter and model parameters if error is less than in previous iteration
    if error < min_error:
        min_error = error
        w = np.copy(w_temp)

#Evaluating the model's performance on test data
temp = np.apply_along_axis(lambda x: np.matmul(x, w), 1, matrix_test)
for k in range(len(temp)):
    if temp[k] > 10:
        temp[k] = 1
    elif temp[k] < -10:
        temp[k] = 0
    else:
        temp[k] = 1 / (1 + math.exp(temp[k] * (-1)))
        if temp[k] > 0.5:
            temp[k] = 1
        else:
            temp[k] = 0
temp = (2 * (df_test.iloc[:, -1:].to_numpy().flatten())) + temp

#Determining the values of the variables for confusion matrix
true_negative = temp.tolist().count(0)
false_positive = temp.tolist().count(1)
false_negative = temp.tolist().count(2)
true_positive = temp.tolist().count(3)

#Displaying the metrics of the model
print("Model successfully trained...")
print("Accuracy: {:.2f}%".format(((true_positive + true_negative) / df_test.shape[0]) * 100))
print("Precision: {:.2f}%".format((true_positive / (true_positive + false_positive)) * 100))
print("Recall: {:.2f}%".format((true_positive / (true_positive + false_negative)) * 100))


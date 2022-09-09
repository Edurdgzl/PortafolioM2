import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent (x, y):
    curr_m, curr_b, curr_loss = np.zeros(x.shape[1]), 0, np.zeros((x.shape[0], ))
    epoch = 10000
    n = len(x)
    alpha = .001
    acum_loss = []

    for i in range(epoch):
        y_pred = x.dot(curr_m) + curr_b
        loss = (1/n) * np.sum(np.power(y-y_pred, 2))
        acum_loss.append(loss)
        if not np.array_equal(loss, curr_loss):
            curr_loss = loss
            md = -(2/n) * (x.T).dot(y - y_pred)
            bd = -(2/n) * sum(y - y_pred)
            curr_m = curr_m - alpha * md
            curr_b = curr_b - alpha * bd
        else:
            break
    print("m: {}, b: {}, loss: {}, epoch: {}".format(curr_m, curr_b, loss, i))
    return curr_m, curr_b, acum_loss
    

df = pd.read_csv('Fish.csv')
df = df.drop(['Species'], axis=1)
print("\n\n\nTESTS\n\n\n")
print("TEST 1")
print("-------------------------------------------------------------------------------------------------------------------")
df_test = df.sample(frac=0.20, random_state=60)
df_train = df.drop(df_test.index)

x_train = df_train.drop(['Weight', 'Length2', 'Length3'], axis=1)
x_train = x_train.to_numpy()
y_train = df_train['Weight'].to_numpy()
m, b, acum_loss = gradient_descent(x_train, y_train)

x_test = df_test.drop(['Weight', 'Length2', 'Length3'], axis=1)
x_test = x_test.to_numpy()
y_test = df_test['Weight'].to_numpy()

result = x_test.dot(m) + b
print("\nDifference between every value of y and the prediction:\n")
print(result - y_test)
print("-------------------------------------------------------------------------------------------------------------------")


""" print("\nTEST 2")
print("-------------------------------------------------------------------------------------------------------------------")
df_test = df.sample(frac=0.20, random_state=55)
df_train = df.drop(df_test.index)

x_train = df_train.drop(['Weight', 'Length2', 'Length3'], axis=1)
x_train = x_train.to_numpy()
y_train = df_train['Weight'].to_numpy()
m, b = gradient_descent(x_train, y_train)

x_test = df_test.drop(['Weight', 'Length2', 'Length3'], axis=1)
x_test = x_test.to_numpy()
y_test = df_test['Weight'].to_numpy()

result = x_test.dot(m) + b
print("\nDifference between every value of y and the prediction:\n")
print(result - y_test)
print("-------------------------------------------------------------------------------------------------------------------")


print("\nTEST 3")
print("-------------------------------------------------------------------------------------------------------------------")

df_test = df.sample(frac=0.20, random_state=50)
df_train = df.drop(df_test.index)

x_train = df_train.drop(['Weight', 'Length2', 'Length3'], axis=1)
x_train = x_train.to_numpy()
y_train = df_train['Weight'].to_numpy()
m, b = gradient_descent(x_train, y_train)

x_test = df_test.drop(['Weight', 'Length2', 'Length3'], axis=1)
x_test = x_test.to_numpy()
y_test = df_test['Weight'].to_numpy()

result = x_test.dot(m) + b
print("\nDifference between every value of y and the prediction:\n")
print(result - y_test)
print("-------------------------------------------------------------------------------------------------------------------")

print("\n\n\nPREDICTIONS\n\n\n")
print("PREDICTION 1")
print("-------------------------------------------------------------------------------------------------------------------")
pred_x1 = 23.2
print("Fish length: {}".format(pred_x1))
pred_x2 = 11.52
print("Fish height: {}".format(pred_x2))
pred_x3 = 4.02
print("Fish width: {}".format(pred_x3))
pred_x4 = np.array([pred_x1, pred_x2, pred_x3])
prediction = pred_x4.dot(m) + b
print("\nThe prediction is: {}\n".format(prediction))
print("-------------------------------------------------------------------------------------------------------------------")

print("\nPREDICTION 2")
print("-------------------------------------------------------------------------------------------------------------------")
pred_x1 = 30.5
print("Fish length: {}".format(pred_x1))
pred_x2 = 15.11
print("Fish height: {}".format(pred_x2))
pred_x3 = 5.2
print("Fish width: {}".format(pred_x3))
pred_x4 = np.array([pred_x1, pred_x2, pred_x3])
prediction = pred_x4.dot(m) + b
print("\nThe prediction is: {}\n".format(prediction))
print("-------------------------------------------------------------------------------------------------------------------")

print("\nPREDICTION 3")
print("-------------------------------------------------------------------------------------------------------------------")
pred_x1 = 32.8
print("Fish length: {}".format(pred_x1))
pred_x2 = 16.51
print("Fish height: {}".format(pred_x2))
pred_x3 = 5.85
print("Fish width: {}".format(pred_x3))
pred_x4 = np.array([pred_x1, pred_x2, pred_x3])
prediction = pred_x4.dot(m) + b
print("\nThe prediction is: {}\n".format(prediction))
print("-------------------------------------------------------------------------------------------------------------------") """
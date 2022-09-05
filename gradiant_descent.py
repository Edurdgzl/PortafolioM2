import numpy as np
import pandas as pd

def gradient_descent (x, y):
    curr_m, curr_b, curr_loss = np.zeros(x.shape[1]), 0, np.zeros((x.shape[0], ))
    epoch = 1000
    n = len(x)
    alpha = .000001

    for i in range(epoch):
        y_pred = x.dot(curr_m) + curr_b
        loss = (1/n) * np.sum(np.power(y-y_pred, 2))
        if not np.array_equal(loss, curr_loss):
            curr_loss = loss
            md = -(2/n) * (x.T).dot(y - y_pred)
            bd = -(2/n) * sum(y - y_pred)
            curr_m = curr_m - alpha * md
            curr_b = curr_b - alpha * bd
        else:
            break
    print("m: {}, b: {}, loss: {}, epoch: {}".format(curr_m, curr_b, loss, i))
    return curr_m, curr_b
    


print("\nTEST 1")
print("-------------------------------------------------------------------------------------------------------------------")
df = pd.read_csv('real-estate.csv')

df_test = df.sample(frac=0.20, random_state=60)
df_train = df.drop(df_test.index)

x_train = df_train.drop(['No', 'X1 transaction date', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area'], axis=1)
x_train = x_train.to_numpy()
y_train = df_train['Y house price of unit area'].to_numpy()
m, b = gradient_descent(x_train, y_train)

x_test = df_test.drop(['No', 'X1 transaction date', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area'], axis=1)
x_test = x_test.to_numpy()
y_test = df_test['Y house price of unit area'].to_numpy()

result = x_test.dot(m) + b
print("\nDifference between every value of y and the prediction:\n")
print(result - y_test)
print("-------------------------------------------------------------------------------------------------------------------")


print("\nTEST 2")
print("-------------------------------------------------------------------------------------------------------------------")
df_test = df.sample(frac=0.20, random_state=55)
df_train = df.drop(df_test.index)

x_train = df_train.drop(['No', 'X1 transaction date', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area'], axis=1)
x_train = x_train.to_numpy()
y_train = df_train['Y house price of unit area'].to_numpy()
m, b = gradient_descent(x_train, y_train)

x_test = df_test.drop(['No', 'X1 transaction date', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area'], axis=1)
x_test = x_test.to_numpy()
y_test = df_test['Y house price of unit area'].to_numpy()

result = x_test.dot(m) + b
print("\nDifference between every value of y and the prediction:\n")
print(result - y_test)
print("-------------------------------------------------------------------------------------------------------------------")


print("\nTEST 3")
print("-------------------------------------------------------------------------------------------------------------------")

df_test = df.sample(frac=0.20, random_state=50)
df_train = df.drop(df_test.index)

x_train = df_train.drop(['No', 'X1 transaction date', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area'], axis=1)
x_train = x_train.to_numpy()
y_train = df_train['Y house price of unit area'].to_numpy()
m, b = gradient_descent(x_train, y_train)

x_test = df_test.drop(['No', 'X1 transaction date', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area'], axis=1)
x_test = x_test.to_numpy()
y_test = df_test['Y house price of unit area'].to_numpy()

result = x_test.dot(m) + b
print("\nDifference between every value of y and the prediction:\n")
print(result - y_test)
print("-------------------------------------------------------------------------------------------------------------------")


print("\nPREDICTION 1")
print("-------------------------------------------------------------------------------------------------------------------")
pred_x1 = float(input("Enter house age: "))
pred_x2 = float(input("Enter house latitude: "))
pred_x3 = float(input("Enter house longitude: "))
pred_x4 = np.array([pred_x1, pred_x2, pred_x3])
prediction = pred_x4.dot(m) + b
print("\nThe prediction is: {}\n".format(prediction))
print("-------------------------------------------------------------------------------------------------------------------")


print("\nPREDICTION 2")
print("-------------------------------------------------------------------------------------------------------------------")
pred2_x1 = float(input("Enter house age: "))
pred2_x2 = float(input("Enter house latitude: "))
pred2_x3 = float(input("Enter house longitude: "))
pred2_x4 = np.array([pred2_x1, pred2_x2, pred2_x3])
prediction2 = pred2_x4.dot(m) + b
print("\nThe prediction is: {}\n".format(prediction2))
print("-------------------------------------------------------------------------------------------------------------------")


print("\nPREDICTION 3")
print("-------------------------------------------------------------------------------------------------------------------")
pred3_x1 = float(input("Enter house age: "))
pred3_x2 = float(input("Enter house latitude: "))
pred3_x3 = float(input("Enter house longitude: "))
pred3_x4 = np.array([pred3_x1, pred3_x2, pred3_x3])
prediction3 = pred3_x4.dot(m) + b
print("\nThe prediction is: {}\n".format(prediction3))
print("-------------------------------------------------------------------------------------------------------------------")
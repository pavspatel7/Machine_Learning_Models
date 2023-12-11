import random
import numpy as np
from layout import *
import AI_Learn
import pandas as pd
import ast

size = 20
dataset = 2500

logistic_reg = AI_Learn.LogisticRegression()
softmax_reg = AI_Learn.SoftMax_regression()

print('---------------------------------------------------')
generate_data( dataSet = dataset, gridSize = size)
print('---------------------------------------------------')

def load_data(FilePath):
    df = pd.read_excel(FilePath)
    df['encoding'] = df['encoding'].apply(ast.literal_eval)
    df['multi_classification'] = df['multi_classification'].apply(ast.literal_eval)
    column_values_encoding                = df['encoding'].tolist()
    column_values_status                  = df['binary_classification'].tolist()
    column_values_multi_classification    = df['multi_classification'].tolist()

    return column_values_encoding, column_values_status, column_values_multi_classification

# def load_data(FilePath, start_row, nrows = None):
#     # Specify the number of rows and columns you want to read, along with the starting row
#     df = pd.read_excel(FilePath, skiprows=range(1, start_row), nrows=nrows)
    
#     # Assuming 'Encoding' and 'Status' are always part of the selected columns
#     df['encoding'] = df['encoding'].apply(ast.literal_eval)
    
#     column_values_encoding = df['encoding'].tolist()
#     column_values_binary_classification = df['binary_classification'].tolist()
#     column_values_multi_classification = df['multi_classification'].tolist()
    
#     return column_values_encoding, column_values_binary_classification, column_values_multi_classification


print(" Start Training and validating")
print("------------------------------------------")
# X_Train, Y_Train, Z_Train = load_data(FilePath = 'DataSets/training_dataSet.xlsx')


# X_Train, Y_Train, temp = load_data(FilePath = 'DataSets/training_dataSet.xlsx')
# X_Test, Y_Test, temp = load_data('DataSets/testing_dataSet.xlsx')

# new_weight, x_axis , y_axis_train , train_accuracy,  y_axis_test  , test_accuracy  = logistic_reg.training_model(X_Train = X_Train, Y_Train = Y_Train, X_Test= X_Test, Y_Test=Y_Test, epoch = 50, learning_rate = 0.15, debug = True, weight_no = 0.025)


# logistic_reg.plot_loss(
#     x_axis_test=x_axis,
#     y_axis_test = y_axis_test,
#     x_axis_train=x_axis,
#     y_axis_train=y_axis_train)

# logistic_reg.plot_loss(
#     x_axis_test=x_axis,
#     y_axis_test = test_accuracy,
#     x_axis_train=x_axis,
#     y_axis_train= train_accuracy)

# print("------------------------------------------")
# print(" Testing Accracy on Unseen Data ")
# print("------------------------------------------")
# avg_loss, avg_accuracy = logistic_reg.test_model(encoding = X_Test, target = Y_Test, weight = new_weight)


X_Train , temp , Y_Train = load_data(FilePath = 'DataSets/training_dataSet.xlsx')
X_Test , temp , Y_Test = load_data(FilePath = 'DataSets/testing_dataSet.xlsx')
w_red, w_blue, w_green, w_yellow = softmax_reg.training_model(X_Train = X_Train, Y_Train = Y_Train, X_Test=X_Test, Y_Test=Y_Test, epoch = 50, learning_rate = 0.15, debug = True, weight_no = 0.025)

# x,y, avg = softmax_reg.testing_model(
#                                      X_Test=X_Test, 
#                                      Y_Test=Y_Test,
#                                      weight_blue=w_blue,
#                                      weight_green=w_green,
#                                      weight_red=w_red,
#                                      weight_yellow=w_yellow
#                                      )



# def training_model(encoding, target, weight, epoch, alpha, lambda_reg, validation_x, validation_y, patience):
#     best_weight = None
#     validate_accuracy = 0
#     best_loss = float('inf')
#     no_improvement_count = 0

#     for e in range(epoch):
#         combined_data = list(zip(encoding, target))
#         random.shuffle(combined_data)
#         encoded_x_values, target_y_values = zip(*combined_data) 
#         each_epoch_loss = []
#         true_count = 0

#         for x, t in zip(encoded_x_values, target_y_values):
#             x = add_noise(np.array(x), 0.01)
#             prediction = sigmoid_function(weight, x)
#             true_count += 1 if classified_values(prediction) == t else 0
#             loss = cross_entropy_loss(t, prediction)
#             each_epoch_loss.append(loss)
#             weight = gradient_descent(x, t, prediction, alpha, weight, lambda_reg)
        
#         avg_loss = sum(each_epoch_loss) / len(each_epoch_loss)
#         avg_accuracy = true_count / len(target)
#         print(f"Epoch {e}: Loss = {avg_loss}, Accuracy = {avg_accuracy}")

#         val_loss, val_accuracy = validate_model(validation_x, validation_y, weight)
#         # print(f"validation loss {val_loss} , accuracy {val_accuracy}")
#         if val_loss < best_loss:
#             best_loss = val_loss
#             validate_accuracy = val_accuracy
#             best_weight = weight
#             no_improvement_count = 0
#         else:
#             no_improvement_count += 1
#             if no_improvement_count >= patience:
#                 print("Early stopping triggered.")
#                 break

#     return best_weight

# def gradient_descent(features, target, prediction, alpha, weight, lambda_reg):
#     gradient = np.dot(features, (target - prediction))
#     new_weights = weight + alpha * gradient - alpha * lambda_reg * weight
#     return new_weights 

# def validate_model(encoding, target, weight):
#     total_loss = 0
#     true_count = 0
#     for x, t in zip(encoding, target):
#         prediction = sigmoid_function(weight, x)
#         true_count += 1 if classified_values(prediction) == t else 0
#         loss = cross_entropy_loss(t, prediction)
#         total_loss += loss
#     avg_loss = total_loss / len(target)
#     avg_accuracy = true_count / len(target)
#     return avg_loss, avg_accuracy

            # val_x, val_y, accuracy = test_model(validation_x, validation_y, weight)
            # if accuracy > validateAccuracy:
            #     validateAccuracy = accuracy
            #     bestWeight = weight
            #     print(bestWeight)
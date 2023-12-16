import random
import numpy as np
import pandas as pd
import ast
from layout import *
import AI_Learn


size = 20
# Enter number of data set to generate
dataset = 5000

def load_data(FilePath, read):
    if read == 1:
        df = pd.read_excel(FilePath)
        df['encoding'] = df['encoding'].apply(ast.literal_eval)
        column_values_encoding                = df['encoding'].tolist()
        column_values_status                  = df['binary_classification'].tolist()        
        return column_values_encoding, column_values_status
    
    if read == 2:
        df = pd.read_excel(FilePath)
        df['encoding'] = df['encoding'].apply(ast.literal_eval)
        df['multi_classification'] = df['multi_classification'].apply(ast.literal_eval)
        column_values_encoding                = df['encoding'].tolist()
        column_values_multi_classification    = df['multi_classification'].tolist()
        return column_values_encoding, column_values_multi_classification


logistic_reg = AI_Learn.LogisticRegression()
softmax_reg = AI_Learn.SoftMax_regression()

# print("***********************************************************")
# print("      Model - 1   LOGISTIC REGRESSION                      ")

# # generate data take parameter of dataset and size of image
# # Enter the ratio of data to be split for training and testing
# generate_data_for_task_1( dataSet = dataset, gridSize = size, test_split_ratio = 0.2)

# # Read data from the training data set 
# X_Train, Y_Train = load_data(FilePath = 'DataSets/training_dataSet_1.xlsx', read=1)

# # Read data from the testing data set
# X_Test, Y_Test = load_data(FilePath = 'DataSets/testing_dataSet_1.xlsx', read=1)

# # Read data from the validation data set
# X_validation, Y_validation = load_data(FilePath = 'DataSets/validation_dataSet_1.xlsx', read=1)

# new_weight, x_axis , y_axis_train , train_accuracy,  y_validation_loss  , x_validation_accuracy , aVA, aTR = logistic_reg.training_model(
#     X_Train = X_Train, 
#     Y_Train = Y_Train, 
#     X_Validation= X_validation, 
#     Y_Validation=Y_validation, 
#     epoch = 50, 
#     learning_rate = 0.25, 
#     debug = True, 
#     weight_no = 0.025, 
#     PrintReport = True,
#     patience = 8
#     )

# test_loss, aTT = logistic_reg.test_model(
#     encoding= X_Test , 
#     target= Y_Test, 
#     weight = new_weight)

# print("                      Testing Report                       ")
# print("***********************************************************")
# print("Testing Data Set Accuracy : ", round(aTT,5))
# print("***********************************************************")

# logistic_reg.plot_loss(
#     x_axis_test=x_axis,
#     y_axis_test = y_validation_loss,
#     test_rep = "Validation Loss",    
#     x_axis_train=x_axis,
#     y_axis_train=y_axis_train,
#     train_rep = "Training Loss",
#     x_label = "Epoch",
#     y_label = "Loss",
#     t_title = f"Close this Graph to Finish excuting Further \n\n\n Training vs Validation Loss at Each Epoch {dataset} Data Sets"
# )

# logistic_reg.plot_loss(
#     x_axis_test=x_axis,
#     y_axis_test = x_validation_accuracy,
#     test_rep = "Validation Accuracy",    
#     x_axis_train=x_axis,
#     y_axis_train= train_accuracy,
#     train_rep = "Training Accuracy",
#     x_label = "Epoch",
#     y_label = "Accuracy",
#     t_title = "Close this Graph to Finish excuting Further \n\n\n Training vs Validation Accuracy at Each Epoch "
# )

# logistic_reg.plot_bar(
#     training= round(aTR,5),
#     validation= round(aVA,5),
#     testing= round(aTT,5)
# )


print("\n\n")
print("***********************************************************")
print("      Model - 2   SOFTMAX REGRESSION                       ")



# Generate Data for the Task - 2
generate_data_for_task_2(dataSet = dataset, gridSize = size, test_split_ratio = 0.2)

# Read Data from the training data set 
X_Train , Y_Train = load_data(FilePath = 'DataSets/training_dataSet_2.xlsx', read=2)

# Read data from the testing data sets
X_Test , Y_Test = load_data(FilePath = 'DataSets/testing_dataSet_2.xlsx', read=2)

# Read data from the validation data sets
X_Validation , Y_Validation = load_data(FilePath = 'DataSets/validation_dataSet_2.xlsx', read=2)

# Training the model using the softmax regression
new_weights, x_axis, y_train_loss, y_train_accuracy, y_validation_loss, y_validation_accuracy, aTR, aVA = softmax_reg.training_model(   
    X_Train = X_Train, 
    Y_Train = Y_Train, 
    X_Validation=X_Validation, 
    Y_Validation=Y_Validation, 
    epoch = 50, 
    learning_rate = 0.0005, 
    debug = True, 
    weight_no = 0.035,
    patience = 5,
    PrintReport = True
)

# Testing the model with the updated weights using the softmax regression
avg_loss, avg_accuracy = softmax_reg.testing_model(X_Test = X_Test, Y_Test = Y_Test, weight = new_weights)


print("                      Testing Report                       ")
print("***********************************************************")
print("Testing Data Set Accuracy : ", round(avg_accuracy,5))
print("***********************************************************")

logistic_reg.plot_loss(
    x_axis_test= x_axis,
    y_axis_test = y_validation_loss,
    test_rep = "Validation Loss",    
    x_axis_train= x_axis,
    y_axis_train= y_train_loss,
    train_rep = "Training Loss",
    x_label = "Epoch",
    y_label = "Loss",
    t_title = f" Close this Graph to Finish excuting Further \n\n\n Training vs Validation Loss at Each Epoch {dataset} Data Sets"
)

logistic_reg.plot_loss(
    x_axis_test=x_axis,
    y_axis_test = y_validation_accuracy,
    test_rep = "Validation Accuracy",    
    x_axis_train=x_axis,
    y_axis_train= y_train_accuracy,
    train_rep = "Training Accuracy",
    x_label = "Epoch",
    y_label = "Accuracy",
    t_title = " Close this Graph to Finish excuting Further \n\n\n Training vs Validation Accuracy at Each Epoch "
)

logistic_reg.plot_bar (
    training = round(aTR,5),
    validation = round(aVA,5),
    testing = round(avg_accuracy,5)       
)
import random
import numpy as np
from layout import *
import AI_Learn
import pandas as pd
import ast

size = 20
dataset = 5000

logistic_reg = AI_Learn.LogisticRegression()

generate_data( dataSet=dataset, gridSize=size , trainingSize=0, testSize=0)


def load_data(FilePath):
    df = pd.read_excel(FilePath)
    df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
    column_values_encoding = df['Encoding'].tolist()
    column_values_status = df['Status'].tolist()
    return column_values_encoding, column_values_status


print(" Start Training and validating")
print("------------------------------------------")
X_Train, Y_Train = load_data('DataSets/encoding_results.xlsx')
X_Test, Y_Test = load_data('DataSets/encoding_test.xlsx')

new_weight, x, y  = logistic_reg.training_model(X_Train = X_Train, Y_Train = Y_Train, epoch=50, learning_rate=0.15, debug=True, weight_no = 0.025)

print("------------------------------------------")
print(" Testing Accracy on Unseen Data ")
print("------------------------------------------")
x,y, avg = logistic_reg.test_model(encoding = X_Test, target = Y_Test, weight = new_weight)



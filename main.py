import random
import numpy as np
from layout import *
from Model_1 import *
from Model_2 import *
import csv
import pandas as pd
import openpyxl
import ast

size = 20
result_list = []

dataset = 1

for i in range(dataset):
    print("-------------------------------------------------")
    layout = creatingLayout()
    grid = layout.wiredGrid(size)
    for x in grid:
        print(''.join(x))
    print()
    status = layout.wiredGrid_status_is()

    result_dict = {'Encoding': encoding(grid), 'Status': status}
    result_list.append(result_dict)

    if i == 499:
        result_df = pd.DataFrame(result_list)
        result_df.to_excel('encoding_validation.xlsx', index=False)
        result_list = []

    if i == 999:
        result_df = pd.DataFrame(result_list)
        result_df.to_excel('encoding_results.xlsx', index=False)
        result_list = []

    if i == 1499:
        result_df = pd.DataFrame(result_list)
        result_df.to_excel('encoding_test.xlsx', index=False)





features = 1601
weight = np.random.uniform(-0.5, 0.5, features)

print(" Start Training and validating")
print("------------------------------------------")
df = pd.read_excel('encoding_results.xlsx')
df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
training_x_value = df['Encoding'].tolist()
training_y_value = df['Status'].tolist()

df = pd.read_excel('encoding_validation.xlsx')
df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
validation_x_value = df['Encoding'].tolist()
validation_y_value = df['Status'].tolist()


# new_weight  = training_model(encoding=training_x_value, target=training_y_value, weight= weight, epoch=50, alpha=0.00000001, validation_x=validation_x_value, validation_y=validation_y_value, patience=7, lambda_reg=0.00000001)

new_weight  = training_model(encoding=training_x_value, target=training_y_value, weight= weight, epoch=50, alpha=0.01, validation_x=validation_x_value, validation_y=validation_y_value)


N_weight = new_weight

df = pd.read_excel('encoding_test.xlsx')
df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
testing_x_value = df['Encoding'].tolist()
testing_y_value = df['Status'].tolist()

print("------------------------------------------")
print(" Testing Accracy on Unseen Data ")
print("------------------------------------------")
x,y, avg = test_model(encoding=testing_x_value, target=testing_y_value, weight=N_weight)



import numpy as np
import pandas as pd
from Model_1 import *
import ast

df = pd.read_excel('encoding_results.xlsx')

df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
# Assuming the column you want to print is named 'Encoding'
column_values_encoding = df['Encoding'].tolist()
column_values_status = df['Status'].tolist()


weight = np.random.uniform(0, 0, features)

print("Training")

df = pd.read_excel('encoding_validation.xlsx')
df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
encoding_v = df['Encoding'].tolist()
status_v = df['Status'].tolist()

weight, x,y , bias = training_model(encoding=column_values_encoding, weight= weight, target=column_values_status, epoch=100, alpha=0.01, validation_x=encoding_v, validation_y=status_v)


N_weight = weight
N_bias = bias

df = pd.read_excel('encoding_test.xlsx')
df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
# Assuming the column you want to print is named 'Encoding'
column_values_encoding = df['Encoding'].tolist()
column_values_status = df['Status'].tolist()

x,y, avg = test_model(encoding=column_values_encoding, target=column_values_status, weight=N_weight, bias= N_bias)



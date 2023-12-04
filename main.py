import random
import numpy as np
from layout import *
from Model_1 import *
from Model_2 import *
import csv
import pandas as pd
import openpyxl

size = 20
result_list = []

def encoding(grid):
    color_mapping = {'⬜️': [1, 0, 0, 0, 0],
                     '🟥': [0, 0, 0, 0, 1],
                     '🟨': [0, 0, 0, 1, 0],
                     '🟦': [0, 0, 1, 0, 0],
                     '🟩': [0, 1, 0, 0, 0]}
    result = []
    for row in grid:
        for cell in row:
            result.extend(color_mapping.get(cell, [0, 0, 0, 0, 0]))
    return result


for i in range(1000):
    print("-------------------------------------------------")
    layout = creatingLayout()
    grid = layout.wiredGrid(size)
    status = layout.wiredGrid_status_is()

    result_dict = {'Encoding': encoding(grid), 'Status': status}
    result_list.append(result_dict)

result_df = pd.DataFrame(result_list)
result_df.to_excel('encoding_results.xlsx', index=False)




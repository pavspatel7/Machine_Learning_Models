import random
import numpy as np
from layout import *
from Model_1 import *
from Model_2 import *
import csv
import pandas as pd
import openpyxl

size = 20
color_mapping = {'⬜️': 1, '🟥': 1, '🟨': 1, '🟦': 1, '🟩': 1}
csv_file_name = 'output.csv'


# for x in grid:
#     print(''.join(x))
# print()

def encoding(grid):
    redc = [0, 0, 0, 0, 1]
    greenc = [0, 0, 0, 1, 0]
    yellowc = [0, 0, 1, 0, 0]
    bluec = [0, 1, 0, 0, 0]
    whitec = [1, 0, 0, 0, 0]
    result = []
    for x in range(len(grid)):
        for y in range(len(grid)):
            if grid[x][y] == '⬜️':
                result.extend(whitec)
            if grid[x][y] == '🟥':
                result.extend(redc)
            if grid[x][y] == '🟨':
                result.extend(yellowc)
            if grid[x][y] == '🟦':
                result.extend(bluec)
            if grid[x][y] == '🟩':
                result.extend(greenc)

    return result

for i in range(30):
    print("-------------------------------------------------")
    layout = creatingLayout()
    grid = layout.wiredGrid(size)
    status = layout.wiredGrid_status_is()
    print("status = ", status)
    print(encoding(grid))
    m = Model1(encoding(grid), 0.01, status)
    m.cross_entropy_loss()

# iteration_df = pd.DataFrame({'Encoding': [encoding(grid)], 'Status': [status]})

# Save the DataFrame to an Excel file
# iteration_df.to_excel('encoding_result_with_data.xlsx', index=False, mode='a', header=not bool(1))

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# r = []
#
# # Writing to CSV file
# with open(csv_file_name, 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#
#     # Write the header row
#     header_row = [f'col_{i+1}' for i in range(0, 401)]
#     csv_writer.writerow(header_row)
#
#     for i in range(500):
#         layout = creatingLayout()
#         r = []
#         try:
#             print("-------------------------------------------------")
#             grid = layout.wiredGrid(size)
#             print("status = ", layout.wiredGrid_status_is())
#
#             for x in grid:
#                 print(''.join(x))
#             print()
#
#             # Write data to the CSV file
#             for x in range(len(grid)):
#                 csv_row = [color_mapping[grid[x][y]] for y in range(len(grid))]
#                 r.extend(csv_row)
#             r.append(layout.wiredGrid_status_is())
#             csv_writer.writerow(r)
#
#         except Exception as e:
#             print(e)
#
#     print(f'Data has been written to {csv_file_name}.')


# def get_random_row(file_name):
#     with open(file_name, 'r') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         rows = list(csv_reader)
#         if len(rows) > 1:  # Ensure there is at least one row (excluding the header)
#             index = random.randint(1, len(rows) - 1)  # Random index (excluding the header row)
#             random_row = rows[index]
#             return index, random_row
#         else:
#             return None
#
# import csv
#
# def get_row_by_index(file_name, row_index):
#     with open(file_name, 'r') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         rows = list(csv_reader)
#         if 1 <= row_index < len(rows):  # Check if the index is within valid range
#             specified_row = rows[row_index]
#             return row_index, specified_row
#         else:
#             return None
#
# result = get_row_by_index('output.csv', 2)

# Example usage
# result = get_random_row(csv_file_name)

# if result:
#     index, random_row = result
#     print(f"Random Row (Index {index}): {random_row}")
#     m = Model1(random_row, 0.01)
#     # m.sigmoid_function()
#     m.cross_entropy_loss()
# else:
#     print("CSV file is empty.")

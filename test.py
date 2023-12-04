import numpy as np
# from Model_1 import *
# # Assuming 400 features (pixels) for simplicity
# num_features = 400
#
# # Initialize weights with small random values from a Gaussian distribution
# # weights = np.random.normal(loc=0, scale=0.01, size=num_features)
# weights = np.random.rand(400)
# print(weights)
#
a = [1,2]
b = [3,4]

print(np.array(a))

# colors = ['⬜️', '🟥', '🟨', '🟦', '🟩']
# encoded_array = np.zeros((20, 20, len(colors)), dtype=int)
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(encoded_array.flatten())
#
# df = pd.read_excel('encoding_results.xlsx')
#
# # Assuming the column you want to print is named 'Encoding'
# column_values_encoding = df['Encoding'].tolist()
# column_values_status = df['Status'].tolist()
#
# for item, s in zip(column_values_encoding, column_values_status):
#     print(item, s)
#
# print(len(column_values_encoding))
# print(len(column_values_status))
# # Initialize bias
# # bias = np.random.rand(1)
# # print(bias)
#
#
#
#
# # m = Model1(encoding(grid), 0.01, status)
# # m.cross_entropy_loss()
#
#
# # iteration_df = pd.DataFrame({'Encoding': [encoding(grid)], 'Status': [status]})
#
# # Save the DataFrame to an Excel file
# # iteration_df.to_excel('encoding_result_with_data.xlsx', index=False, mode='a', header=not bool(1))
#
# # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
#
# # r = []
# #
# # # Writing to CSV file
# # with open(csv_file_name, 'w', newline='') as csvfile:
# #     csv_writer = csv.writer(csvfile)
# #
# #     # Write the header row
# #     header_row = [f'col_{i+1}' for i in range(0, 401)]
# #     csv_writer.writerow(header_row)
# #
# #     for i in range(500):
# #         layout = creatingLayout()
# #         r = []
# #         try:
# #             print("-------------------------------------------------")
# #             grid = layout.wiredGrid(size)
# #             print("status = ", layout.wiredGrid_status_is())
# #
# #             for x in grid:
# #                 print(''.join(x))
# #             print()
# #
# #             # Write data to the CSV file
# #             for x in range(len(grid)):
# #                 csv_row = [color_mapping[grid[x][y]] for y in range(len(grid))]
# #                 r.extend(csv_row)
# #             r.append(layout.wiredGrid_status_is())
# #             csv_writer.writerow(r)
# #
# #         except Exception as e:
# #             print(e)
# #
# #     print(f'Data has been written to {csv_file_name}.')
#
#
# # def get_random_row(file_name):
# #     with open(file_name, 'r') as csvfile:
# #         csv_reader = csv.reader(csvfile)
# #         rows = list(csv_reader)
# #         if len(rows) > 1:  # Ensure there is at least one row (excluding the header)
# #             index = random.randint(1, len(rows) - 1)  # Random index (excluding the header row)
# #             random_row = rows[index]
# #             return index, random_row
# #         else:
# #             return None
# #
# # import csv
# #
# # def get_row_by_index(file_name, row_index):
# #     with open(file_name, 'r') as csvfile:
# #         csv_reader = csv.reader(csvfile)
# #         rows = list(csv_reader)
# #         if 1 <= row_index < len(rows):  # Check if the index is within valid range
# #             specified_row = rows[row_index]
# #             return row_index, specified_row
# #         else:
# #             return None
# #
# # result = get_row_by_index('output.csv', 2)
#
# # Example usage
# # result = get_random_row(csv_file_name)
#
# # if result:
# #     index, random_row = result
# #     print(f"Random Row (Index {index}): {random_row}")
# #     m = Model1(random_row, 0.01)
# #     # m.sigmoid_function()
# #     m.cross_entropy_loss()
# # else:
# #     print("CSV file is empty.")


import numpy as np


# Activation Function
def sigmoid(w_sum):
    return 1 / (1 + np.exp(-w_sum))


# Get Prediction
def predict(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)


# Loss Function
def cross_entropy(target, pred):
    return -(target * np.log10(pred) + (1 - target) * (np.log10(1 - pred)))


# Update Weights
def gradient_descent(x, y, weights, bias, learnrate, pred):
    new_weights = []
    bias += learnrate * (y - pred)

    for w, xi in zip(weights, x):
        new_weight = w + learnrate * (y - pred) * xi
        new_weights.append(new_weight)
    return new_weights, bias


# Data
features = np.array(([0.1, 0.5, 0.2], [0.2, 0.3, 0.1], [0.7, 0.4, 0.2], [0.1, 0.4, 0.3]))
targets = np.array([0, 1, 0, 1])

epochs = 10
learnrate = 0.1

errors = []
weights = np.array([0.4, 0.2, 0.6])
bias = 0.5

new_weights = []

for e in range(epochs):
    for x, y in zip(features, targets):
        pred = predict(x, weights, bias)
        error = cross_entropy(y, pred)
        weights, bias = gradient_descent(x, y, weights, bias, learnrate, pred)

    # Printing out the log-loss error on the training set
    out = predict(features, weights, bias)
    loss = np.mean(cross_entropy(targets, out))
    errors.append(loss)
    print("\n========== Epoch", e, "==========")
    print("Average loss: ", loss)
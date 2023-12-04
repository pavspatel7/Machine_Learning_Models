import numpy as np
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt

features = 2000
# weight = np.random.uniform(-0.5, 0.5, features)
bias = 0.5
epoch = 500


def training_model(encoding, weight, target, epoch, alpha):
    # global weight, bias
    x_axis = []
    y_axis = []
    for e in range(epoch):
        each_epoch_loss = []
        for x, s in zip(encoding, target):
            prediction = sigmoid_function(weight, x, bias)
            loss = cross_entropy_loss(s, prediction)
            each_epoch_loss.append(loss)
            weight = gradient_descent(x, s, prediction, alpha, bias, weight)
        x_axis.append(e)
        y_axis.append(sum(each_epoch_loss) / len(each_epoch_loss))
        print(f"Loss: {y_axis[e]} at time {e}")
    plt.plot(x_axis, y_axis)
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Loss at each time training 500 images')
    # Displaying the graph
    plt.show()
    return weight


def logistic_model(encoding, target, weight):
    for x, s in zip(encoding, target):
        prediction = sigmoid_function(weight, x, bias)
        loss = cross_entropy_loss(s, prediction)
        print(f"Prediction: {prediction: <20} Loss: {loss: 10}")


def sigmoid_function(weight, x_vector, bias):
    weight_sum = np.dot(weight, x_vector) + bias
    predicted_value = 1 / (1 + np.exp((-weight_sum)))
    return predicted_value


def cross_entropy_loss(targeted_value, predicted_value):
    loss = (-1 * float(targeted_value) * np.log(predicted_value)) - (
                float((1 - targeted_value)) * np.log(1 - predicted_value))
    return loss


def gradient_descent(features, target, prediction, alpha, bias, weight):
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    new_weights = []
    # new_bias = bias + alpha * (target - prediction)
    for x, w in zip(features, weight):
        new_w = w + alpha * (target - prediction) * float(x)
        new_weights.append(new_w)
    return new_weights  # , new_bias


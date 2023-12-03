import numpy as np
import random

class Model1():

    def __init__(self, givenInput_Array, givenAlpha, target):
        self.predicted_value = None
        self.alpha = givenAlpha
        self.x_vector = givenInput_Array
        self.targeted_value = target
        print("targeted", self.targeted_value)
        # self.epoch = 400
        self.bias = 0.5
        self.weight = np.random.uniform(-0.5,0.5,(len(givenInput_Array)))

    def sigmoid_function(self):
        try:
            weight_sum = sum(map(lambda x, y: x * float(y), self.weight, self.x_vector))
            weight_sum += self.bias
            print("weight_sum",weight_sum)
            self.predicted_value = 1 / (1 + np.exp( (-weight_sum) ))
            print("prediction", self.predicted_value)
        except Exception as e:
            print(e)
            return 0

        return self.predicted_value

    def cross_entropy_loss(self):
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # print("Old weight: ",len(self.weight.flatten()))
        try:
            self.predicted_value = self.sigmoid_function() # - 0.01
            loss = (-1 * self.targeted_value * np.log(self.predicted_value)) - ((1 - self.targeted_value) * np.log(1 - self.predicted_value))
            print("loss", loss)

            we, ba = self.gradient_descent(self.x_vector, self.targeted_value, self.predicted_value, self.alpha, self.bias)

            print("New Bias: ", ba)
            print("New Weights: ", we)

            print("----------------------------------------")
        except Exception as e:
            print(e)
            return 0
        return loss

    # new_weight = old_weight + alpha * (target - pridction) * x[]
    # new_bias = old_bias + alpha * (target - pridction)

    def gradient_descent(self, features, target, prediction, alpha, bias):
        new_weights = []
        new_bias = bias + alpha * (target - prediction)
        for x,w in zip(features, self.weight):
            new_w = w + alpha * (target - prediction) * float(x)
            new_weights.append(new_w)

        return new_weights, new_bias

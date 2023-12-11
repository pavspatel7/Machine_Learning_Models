import numpy as np
import random
import matplotlib.pyplot as plt
import ast
import math

# Colors One Hot Encoding dictionary

color_mapping = {
                '⬜️': [ 0, 0, 0, 0 ],
                '🟥': [ 0, 0, 0, 1 ],
                '🟨': [ 0, 0, 1, 0 ],
                '🟦': [ 0, 1, 0, 0 ],
                '🟩': [ 1, 0, 0, 0 ]    
                }

# Image Pixel into One Hot Encoding
def encoding(grid):
    global color_mapping
    # Storing all encodings into result list
    result = []
    # First value of vector to be 1
    result.append(1)
    # iterating through grid and encoding ecah pixel (cell)
    for row in grid:
        for cell in row:
            # To get an 1D array using extend
            result.extend(color_mapping.get(cell, [ 0, 0, 0, 0]))
    return result


def order_encoding(color_order):
    global color_mapping
    binary = []
    get_color  = color_order[2][0]
    binary.extend(color_mapping.get(get_color, [ 0, 0, 0, 0]))
    return binary
    


class LogisticRegression():

    """
    Training Model:
        It takes parameters of X-train, Y-train, X-test, Y-test, number of epoch,
        learning rate, debug,

        Implementations:

    """
    def training_model(self, X_Train, Y_Train, X_Test, Y_Test, epoch, learning_rate, weight_no  , debug):
        
        x_axis = []
        y_axis_train = []
        y_axis_test = []
        test_accuracy = []
        train_accuracy  = []
        weight = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))

        # Training Data for each epoch
        for e in range(epoch):

            # Create variable containes X vector and Y vector of Training Data Set. 
            combined_data = list(zip(X_Train, Y_Train))
            # Perform random Shuffle in combined data
            # Avoiding training model in sequence
            random.shuffle(combined_data)
            # unzip Data 
            encoding_shuffled, target_shuffled = zip(*combined_data) 

            # Creating an empty list at each epoch to store current loss
            each_epoch_loss = []
            # Counting number of correct prediction from the training data set
            correct_prediction_count = 0
            train_accuracy_epoch = 0
            

            print("--------------------------------------------------------------")

            for x, t in zip(encoding_shuffled, target_shuffled):

                # Calculating the current data point prediction using sigmoid function
                prediction = self.sigmoid_function(weight, x)
                # Finding the possiblity of correct prediction on the given data point
                correct_prediction_count += 1  if self.classified_values(prediction) == t else 0
                # Calculate the loss on the data point of X vector, of prediction from target
                loss = self.cross_entropy_loss(t, prediction)
                # Appending each loss training data set, of each epoch
                each_epoch_loss.append(loss)
                # Updating Weight using the Stochastic gradient descent
                weight = self.gradient_descent(x, t, prediction, learning_rate, weight)
            
                
            # Calculate accuracy of correct prediction of training set
            train_accuracy_epoch  = correct_prediction_count / len(Y_Train)
            train_accuracy.append(correct_prediction_count / len(Y_Train))
            
            # Appending number of epochs to x_axis values
            x_axis.append(e)
            # Appending number of average loss for each epoch. 
            y_axis_train.append( sum(each_epoch_loss) / len(each_epoch_loss))
            
            test_loss, temp_test_accuracy = self.test_model(encoding=X_Test , target= Y_Test, weight = weight)
            
            test_accuracy.append(temp_test_accuracy)
            y_axis_test.append(test_loss)
            
            if debug:
                print(f"Loss: { float( y_axis_train[e]) } at time {e}  Accuracy: {train_accuracy_epoch }")
            
            
            
        return weight, x_axis , y_axis_train , train_accuracy,  y_axis_test  , test_accuracy

    # Classiication of prediction
    def classified_values(self, prediction):
        # If Prediction value is greater than 0.5 then return 1
        if prediction > 0.5:
            # return 1 - Dangerous
            return 1
        else:
            # return 0 - Safe
            return 0

    # Testing Model Method
    def test_model(self, encoding, target, weight):
        y_axis_test = []
        # x_axis_test = []
        i = 0
        TrueCount = 0

        # Iterating through zip of X-Test, and Y-Test
        for x, s in zip(encoding, target):
            # Calculating the current data point prediction using sigmoid function
            prediction = self.sigmoid_function(weight, x)
            # Finding the possiblity of correct prediction on the given data point
            TrueCount += 1  if self.classified_values(prediction) == s else 0
            # Calculate the loss on the data point of X vector, of prediction from target
            loss = self.cross_entropy_loss(s, prediction)

            y_axis_test.append(loss)
            # x_axis_test.append(i)
            i+=1

        avg_loss = sum(y_axis_test) / len(y_axis_test)
        avg_accuracy = TrueCount / len(target)
        print(f"Accuracy for Testing Set {avg_accuracy}")
        return avg_loss, avg_accuracy

    # Sigmoid Function: 
    def sigmoid_function(self, weight, x_vector):
        # Taking a dot product between two vectors weight and X-vector
        weight_sum = np.dot(weight, x_vector)
        # calculating prediction using the dot product of weight and x-vector
        # Added 0.00001 to ignore the pridiction values are equal to 0
        predicted_value = ( 1 / (1 + np.exp((-weight_sum)))) + 0.00001
        # return prediction value of 8 decimals
        return round(predicted_value, 8)

    # Loss Function: 
    def cross_entropy_loss(self, target_value, predicted_value):
        # Checking if the prediction value is greater than 1.0 becuase of adding 0.00001 in sigmoid function
        if predicted_value >= 1.0:
            # Considering prediction is almost 1
            predicted_value = 0.99999999
        # Checking if the prediction is less than 0.0
        if predicted_value <= 0.0:
            # Considering the predicton is almost 0 by prediction is equal to 0.00000001
            predicted_value = 0.00000001
        
        # Calculating Loss using actual output of the given input x-vector , and
        # model predicted value for the given x-vector. 
        loss = (-1 * float(target_value) * np.log( predicted_value  )) - (
                    float((1 - target_value)) * np.log(1 - predicted_value))
        return loss

    # Stochastic Gradient Descent:
    # It take input parameter of 
    def gradient_descent(self, X_vector, target_value, predicted_value, alpha, weight):
        # Calculating Error between the difference of target value and predicted value
        error = (target_value-predicted_value)
        # Updating weighs based on the learning rate and the error
        # Taking a dot product of X_vector with the error of prediction. 
        new_weights = weight + alpha * np.dot(X_vector, error)
        return new_weights 

    def add_noise(self, one_hot_vector, noise_level):
        # Generate random noise with the same shape as the input vector
        noise = np.random.uniform(low=-noise_level, high=noise_level, size=one_hot_vector.shape)
        # Add noise to the one-hot vector
        noisy_vector = one_hot_vector + noise
        # Clip values to ensure they remain within [0, 1]
        noisy_vector = np.clip(noisy_vector, 0, 1)
        # Ensure the vector remains normalized (sums to 1)
        noisy_vector /= np.sum(noisy_vector)
        return noisy_vector


    def plot_loss(self, x_axis_train, y_axis_train, x_axis_test, y_axis_test, ):
        plt.plot(x_axis_test, y_axis_test, label="Testing Loss")
        plt.plot(x_axis_train, y_axis_train, label="Training Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Testing Loss')
        plt.legend()
        plt.show()


















class SoftMax_regression():
    
    def training_model(self, X_Train, Y_Train, X_Test, Y_Test, epoch, learning_rate, weight_no, debug):
        
        #GBYR
        
        weight_yellow = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        weight_green = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        weight_red = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        weight_blue = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        # decay_rate = 0.1
        for e in range(epoch):
            
            # learning_rate = learning_rate / (1 + epoch * decay_rate)
            
            # Create variable containes X vector and Y vector of Training Data Set. 
            combined_data = list(zip(X_Train, Y_Train))
            # Perform random Shuffle in combined data
            # Avoiding training model in sequence
            random.shuffle(combined_data)
            # unzip Data 
            encoding_shuffled, target_shuffled = zip(*combined_data)
            each_epoch_loss = []
            correct_prediction_count = 0
            # Calculating acurracy of correct prediction out of given training set
            accuracy_of_correct_prediction  = 0
            
            for x_vector , y_vector in zip( encoding_shuffled, target_shuffled ):
                
                prediction_list = self.softmax_function(weight_blue, weight_yellow, weight_green, weight_red , x_vector )
                correct_prediction_count += 1  if np.argmax(prediction_list) == np.argmax(y_vector) else 0
                loss = self.cross_entropy_loss( prediction_list , y_vector)
                each_epoch_loss.append(loss)
                weight_green , weight_blue , weight_yellow, weight_red = self.gradient_descent(weight_blue, weight_yellow, weight_green, weight_red, x_vector, learning_rate, prediction_list, y_vector)

            accuracy_of_correct_prediction = correct_prediction_count / len(X_Train)
            
            a,b, accuracy_testing = self.testing_model(X_Test=X_Test, Y_Test=Y_Test, weight_blue= weight_blue, weight_green=weight_green, weight_red=weight_red, weight_yellow=weight_yellow)
            
            if debug:
                print(F" Loss: { sum(each_epoch_loss) / len(each_epoch_loss)} time: {e} Accuracy of training: {accuracy_of_correct_prediction} ||| Accuracy of Testing {accuracy_testing}")
            
            # if e == 2:
            #     break
            
        return weight_red , weight_blue , weight_green, weight_yellow


    def classified_values(self, prediction):
        pass
        

    def testing_model(self, X_Test, Y_Test, weight_red , weight_blue , weight_green, weight_yellow):
        new_y = []
        new_x = []
        i = 0
        correct_prediction_count = 0

        # Iterating through zip of X-Test, and Y-Test
        for x_vector, y_vector in zip(X_Test, Y_Test):
            prediction_list = self.softmax_function(weight_blue, weight_yellow, weight_green, weight_red , x_vector )
            correct_prediction_count += 1  if np.argmax(prediction_list) == np.argmax(y_vector) else 0
            loss = self.cross_entropy_loss( prediction_list , y_vector)
            new_y.append(loss)
            new_x.append(i)
            i+=1

        avg = correct_prediction_count / len(Y_Test)
        # print(f"Accuracy for Testing Set {avg}")
        return new_x, new_y, avg

    def cal_z(self, weight, x_vector):
        return np.dot(weight, x_vector)
    
    def softmax_function(self, WB, WY, WG, WR, x_vector):
        # Calculating dot product of red wire weight and x vector
        exp_red = np.exp(self.cal_z(WR, x_vector))

        # Calculating dot product of green wire weight and x vector
        exp_green = np.exp(self.cal_z(WG, x_vector))

        # Calculating dot product of blue wire weight and x vector
        exp_blue = np.exp(self.cal_z(WB, x_vector))

        # Calculating dot product of yellow wire weight and x vector
        exp_yellow = np.exp(self.cal_z(WY, x_vector))

        # Calculating total sum of all wire
        Total_sum = exp_red + exp_blue + exp_green + exp_yellow

        function_green = float(exp_green / Total_sum)
        function_blue = float(exp_blue / Total_sum)
        function_yellow = float(exp_yellow / Total_sum)
        function_red = float(exp_red / Total_sum)

        
        return [function_green, function_blue, function_yellow, function_red]

    def cross_entropy_loss(self, predicted_values, target_values):
        loss = -np.sum(target_values * np.log(predicted_values))
        # print("Prediction:- ",predicted_values)
        # print("Target    :-", target_values)
        # print("Loss      :-", loss)
        # print()
        return loss

    def gradient_descent(self, weight_blue, weight_yellow, weight_green, weight_red, x_vector, learning_rate, predicted_values, target):
        # GBYR

        error = np.array(predicted_values) - np.array(target)

        x_vector = np.array(x_vector)

        # Update weights for each class
        new_weight_green = weight_green - (learning_rate * x_vector * error[0])
        new_weight_blue = weight_blue - (learning_rate * x_vector * error[1])
        new_weight_yellow = weight_yellow - (learning_rate * x_vector * error[2])
        new_weight_red = weight_red - (learning_rate * x_vector * error[3])

        return new_weight_green, new_weight_blue, new_weight_yellow, new_weight_red






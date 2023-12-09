import numpy as np
import random
import matplotlib.pyplot as plt

# weight = np.random.uniform(-0.5, 0.5, features)


class LogisticRegression():

    # Image Pixel into One Hot Encoding
    def encoding(self, grid):
        # Colors One Hot Encoding dictionary
        color_mapping = {
                        
                        '⬜️': [ 0, 0, 0, 0 ],
                        '🟥': [ 0, 0, 0, 1 ],
                        '🟨': [ 0, 0, 1, 0 ],
                        '🟦': [ 0, 1, 0, 0 ],
                        '🟩': [ 1, 0, 0, 0 ]
                        
                        }
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

    """
    Training Model:
        It takes parameters of X-train, Y-train, X-test, Y-test, number of epoch,
        learning rate, debug,

        Implementations:

    """
    def training_model(self, X_Train, Y_Train, epoch, learning_rate, weight_no  , debug):
        
        x_axis = []
        y_axis = []
        weight = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        bestWeight = []
        validateAccuracy = 0

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
            # Calculating acurracy of correct prediction out of given training set
            accuracy_of_correct_prediction  = 0

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
            accuracy_of_correct_prediction  = correct_prediction_count / len(Y_Train)
            # Appending number of epochs to x_axis values
            x_axis.append(e)
            # Appending number of average loss for each epoch. 
            y_axis.append(sum(each_epoch_loss) / len(each_epoch_loss))

            if debug:
                print(f"Loss: { float( y_axis[e]) } at time {e}  Accuracy: {accuracy_of_correct_prediction }")
            
        return weight, x_axis, y_axis

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
        new_y = []
        new_x = []
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

            new_y.append(loss)
            new_x.append(i)
            i+=1

        avg = TrueCount / len(target)
        print(f"Accuracy for Testing Set {avg}")
        return new_x, new_y, avg

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


    def plot_loss(self, x_axis_train, y_axis_train, x_axis_test, y_axis_test):
        plt.plot(x_axis_test, y_axis_test, label="Testing Loss")
        plt.plot(x_axis_train, y_axis_train, label="Training Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Testing Loss')
        plt.legend()
        plt.show()



class SoftMax_regression():

    def training_model(self):
        pass

    def testing_model(self):
        pass

    def sigmoid_function(self):
        pass

    def cross_entropy_loss(self):
        pass

    def gradient_descent(self):
        pass





# def training_model(encoding, target, weight, epoch, alpha, lambda_reg, validation_x, validation_y, patience):
#     best_weight = None
#     validate_accuracy = 0
#     best_loss = float('inf')
#     no_improvement_count = 0

#     for e in range(epoch):
#         combined_data = list(zip(encoding, target))
#         random.shuffle(combined_data)
#         encoded_x_values, target_y_values = zip(*combined_data) 
#         each_epoch_loss = []
#         true_count = 0

#         for x, t in zip(encoded_x_values, target_y_values):
#             x = add_noise(np.array(x), 0.01)
#             prediction = sigmoid_function(weight, x)
#             true_count += 1 if classified_values(prediction) == t else 0
#             loss = cross_entropy_loss(t, prediction)
#             each_epoch_loss.append(loss)
#             weight = gradient_descent(x, t, prediction, alpha, weight, lambda_reg)
        
#         avg_loss = sum(each_epoch_loss) / len(each_epoch_loss)
#         avg_accuracy = true_count / len(target)
#         print(f"Epoch {e}: Loss = {avg_loss}, Accuracy = {avg_accuracy}")

#         val_loss, val_accuracy = validate_model(validation_x, validation_y, weight)
#         # print(f"validation loss {val_loss} , accuracy {val_accuracy}")
#         if val_loss < best_loss:
#             best_loss = val_loss
#             validate_accuracy = val_accuracy
#             best_weight = weight
#             no_improvement_count = 0
#         else:
#             no_improvement_count += 1
#             if no_improvement_count >= patience:
#                 print("Early stopping triggered.")
#                 break

#     return best_weight

# def gradient_descent(features, target, prediction, alpha, weight, lambda_reg):
#     gradient = np.dot(features, (target - prediction))
#     new_weights = weight + alpha * gradient - alpha * lambda_reg * weight
#     return new_weights 

# def validate_model(encoding, target, weight):
#     total_loss = 0
#     true_count = 0
#     for x, t in zip(encoding, target):
#         prediction = sigmoid_function(weight, x)
#         true_count += 1 if classified_values(prediction) == t else 0
#         loss = cross_entropy_loss(t, prediction)
#         total_loss += loss
#     avg_loss = total_loss / len(target)
#     avg_accuracy = true_count / len(target)
#     return avg_loss, avg_accuracy

            # val_x, val_y, accuracy = test_model(validation_x, validation_y, weight)
            # if accuracy > validateAccuracy:
            #     validateAccuracy = accuracy
            #     bestWeight = weight
            #     print(bestWeight)
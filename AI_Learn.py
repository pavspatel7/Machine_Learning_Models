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

def mapping(cell):
    return color_mapping.get(cell, [ 0, 0, 0, 0])

# Image Pixel into One Hot Encoding for Model - 2
def encoding_for_1(grid):
    global color_mapping
    # Storing all encodings into result list
    result = []
    # First value of vector to be 1
    result.append(1)
    # iterating through grid and encoding ecah pixel (cell)
    # Linear Features
    for row in grid:
        for cell in row:
            # To get an 1D array using extend
            result.extend(color_mapping.get(cell, [ 0, 0, 0, 0]))
    # print(len(result))
    
    # Non-Linear Features
    section = [0,4,8,12,16]
    for y in range(len(grid[0])):
        for x in section:
            a = color_mapping.get( grid[x][y] , [0,0,0,0])
            b = color_mapping.get( grid[x+1][y] , [0,0,0,0])
            c = color_mapping.get( grid[x+2][y] , [0,0,0,0])
            d = color_mapping.get( grid[x+3][y] , [0,0,0,0])
            result.append(np.dot(np.dot(np.dot(a, b), c), d))        
    
    for x in range(len(grid[0])):
        for y in section:
            a = color_mapping.get( grid[x][y] , [0,0,0,0])
            b = color_mapping.get( grid[x][y+1] , [0,0,0,0])
            c = color_mapping.get( grid[x][y+2] , [0,0,0,0])
            d = color_mapping.get( grid[x][y+3] , [0,0,0,0])
            result.append(np.dot(np.dot(np.dot(a, b), c), d)) 
            
    return result

# Image Pixel into One Hot Encoding for the Model - 2
def encoding_for_2(grid):
    # Initialize vectors for each color
    feature_vector = []
    R_vector = []
    B_vector = []
    G_vector = []
    Y_vector = []
    feature_vector.append(1)
    
    def get_row(i):
        r_count = 0
        y_count = 0
        b_count = 0
        g_count = 0
        color_to_ret = ''
        color_vector = [0]
        for j in range(len(grid[0])):
            pixel = grid[i][j]
            if pixel == '🟥':
                color_vector.extend(color_mapping.get(pixel, [0, 0, 0, 0]))
                r_count += 1
                if r_count > 1:
                    color_to_ret = '🟥'
            elif pixel == '🟨':
                color_vector.extend(color_mapping.get(pixel, [0, 0, 0, 0]))
                y_count += 1
                if y_count > 1:
                    color_to_ret = '🟨'
            elif pixel == '🟦':
                color_vector.extend(color_mapping.get(pixel, [0, 0, 0, 0]))
                b_count += 1
                if b_count > 1:
                    color_to_ret = '🟦'
            elif pixel == '🟩':
                color_vector.extend(color_mapping.get(pixel, [0, 0, 0, 0]))
                g_count += 1
                if g_count > 1:
                    color_to_ret = '🟩'
        return color_to_ret, color_vector
    
    def get_col(i):
        r_count = 0
        y_count = 0
        b_count = 0
        color_vector = [1]
        g_count = 0
        color_to_ret = ''
        for j in range(len(grid[0])):
            pixel = grid[j][i]
            if pixel == '🟥':
                color_vector.extend(color_mapping.get(pixel, [0, 0, 0, 0]))
                r_count += 1
                if r_count > 1:
                    color_to_ret = '🟥'
            elif pixel == '🟨':
                color_vector.extend(color_mapping.get(pixel, [0, 0, 0, 0]))
                y_count += 1
                if y_count > 1:
                    color_to_ret = '🟨'
            elif pixel == '🟦':
                color_vector.extend(color_mapping.get(pixel, [0, 0, 0, 0]))
                b_count += 1
                if b_count > 1:
                    color_to_ret = '🟦'
            elif pixel == '🟩':
                color_vector.extend(color_mapping.get(pixel, [0, 0, 0, 0]))
                g_count += 1
                if g_count > 1:
                    color_to_ret = '🟩'
        return color_to_ret, color_vector
    
    
    # Scan the grid to fill in the vectors
    for i in range(len(grid[0])):
        color = grid[0][i]
        if color != '⬜️':
            color_ret, vector = get_col(i)
            if color_ret == '🟥':
                R_vector = vector
            elif color_ret == '🟦':
                B_vector = vector
            elif color_ret == '🟩':
                G_vector = vector
            elif color_ret == '🟨':
                Y_vector = vector
        else:
            feature_vector.extend(color_mapping.get(color, [0, 0, 0, 0]))
    
    for i in range(len(grid[0])):
        color = grid[i][0]
        if color != '⬜️':
            color_ret, vector = get_row(i)
            if color_ret == '🟥':
                R_vector = vector
            elif color_ret == '🟦':
                B_vector = vector
            elif color_ret == '🟩':
                G_vector = vector
            elif color_ret == '🟨':
                Y_vector = vector
        else:
            feature_vector.extend(color_mapping.get(color, [0, 0, 0, 0]))
        
    # Compute XOR combinations
    RB_xor = [a ^ b for a, b in zip(R_vector, B_vector)]
    RG_xor = [a ^ b for a, b in zip(R_vector, G_vector)]
    RY_xor = [a ^ b for a, b in zip(R_vector, Y_vector)]
    BG_xor = [a ^ b for a, b in zip(B_vector, G_vector)]
    BY_xor = [a ^ b for a, b in zip(B_vector, Y_vector)]
    GY_xor = [a ^ b for a, b in zip(G_vector, Y_vector)]

    # Concatenate all Linear vectors
    # for row in grid:
    #     for cell in row:
    #         # To get an 1D array using extend
    #         feature_vector.extend(color_mapping.get(cell, [ 0, 0, 0, 0]))
            
    feature_vector.extend(R_vector + B_vector + G_vector + Y_vector + RB_xor + RG_xor + RY_xor + BG_xor + BY_xor + GY_xor)

    return feature_vector

# Get the OutPut Space for the Model - 2 third wire into encoding 
def order_encoding(color_order):
    global color_mapping
    binary = []
    get_color  = color_order[2][0]
    binary.extend(color_mapping.get(get_color, [ 0, 0, 0, 0]))
    return binary
    
# IMplementation of the Logistic regression
class LogisticRegression():

    """
    Training Model:
        It takes parameters of X-train, Y-train, X-test, Y-test, number of epoch,
        learning rate, debug,

        Implementations:
    """
    def training_model(self, X_Train, Y_Train, X_Validation, Y_Validation, epoch, learning_rate, weight_no  , debug, PrintReport, patience):
        
        # Variables 
        x_axis = []
        y_axis_train = []
        y_axis_vali = []
        v_accuracy = []
        train_accuracy  = []
        best_weight = None
        validate_accuracy = 0
        best_loss = float('inf')
        
        # Initializing weights with the size of input space data point
        weight = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        print("               Training Model                            ")
        # Training Data for each epoch
        for e in range(epoch):

            # Create variable containes X vector and Y vector of Training Data Set. 
            combined_data = list(zip(X_Train, Y_Train))
            # Perform random Shuffle in combined data
            # Avoiding training model in sequence
            random.shuffle(combined_data)
            # unzip Data 
            encoding_shuffled, target_shuffled = zip(*combined_data) 

            # selected_encoding = encoding_shuffled[:64]
            # selected_target = target_shuffled[:64]
            # Creating an empty list at each epoch to store current loss
            each_epoch_loss = []
            # Counting number of correct prediction from the training data set
            correct_prediction_count = 0
            train_accuracy_epoch = 0

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
            
            validation_loss, validation_accuracy = self.test_model(encoding=X_Validation , target= Y_Validation, weight = weight)
            
            v_accuracy.append(validation_accuracy)
            y_axis_vali.append(validation_loss)
            
            if debug:
                # print(f"Loss: { round(float( y_axis_train[e]),5) } at time {e}  Accuracy: {round(train_accuracy_epoch - 1e-5 , 5) }")
                print(f"Loss: {round(float(y_axis_train[e]), 5):<10} at time {e:<3} Accuracy: {round(train_accuracy_epoch - 1e-5, 5):<10}")
            if validation_loss < best_loss:
                best_loss = validation_loss
                validate_accuracy = validation_accuracy
                best_weight = weight
                no_improvement_count = 0
                if float(y_axis_train[e]) == 1.0:
                    break
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print("Early stopping triggered.")
                    break
        
        if PrintReport:        
            print("***********************************************************")
            print("    LOGISTIC REGRESSION MODEL TRAINING REPORT          ")
            print("***********************************************************")
            print("Accuracy for Training data Set   : ", round(train_accuracy_epoch- 1e-5, 5))
            print("Accuracy for Validation data Set : ", round(validate_accuracy,5))
            print("***********************************************************")
            print()
        
        return best_weight, x_axis , y_axis_train , train_accuracy,  y_axis_vali  , v_accuracy , validate_accuracy, (train_accuracy_epoch - 1e-5)

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

    def plot_loss(self, x_axis_train, y_axis_train, x_axis_test, y_axis_test, x_label, y_label, t_title, test_rep, train_rep ):
        plt.plot(x_axis_test, y_axis_test, label= test_rep)
        plt.plot(x_axis_train, y_axis_train, label= train_rep)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(t_title)
        plt.legend()
        plt.show()

    def plot_bar(self, training, validation, testing):
        plt.plot(training, label= "training")
        plt.plot(validation, label= "validation")
        plt.plot(testing, label= "testing")
        plt.legend()
        plt.show()

    def plot_bar(self, training, validation, testing):
        # Create a bar plot with training, validation, and testing values
        categories = ['Training', 'Validation', 'Testing']
        values = [training, validation, testing]
        plt.bar(categories, values, color=['red', 'grey', 'pink'],width= 0.5)
        plt.xlabel('Phase')
        plt.ylabel('Accuracy')
        plt.title(' Close this Graph to Finish excuting Further \n\n\n Accuracy Benchmarks')
        for i, value in enumerate(values):
            plt.text(i, value, str(value), ha='center', va='bottom')
        plt.show()

# IMplementation of the Softmax regression
class SoftMax_regression():

    def training_model(self, X_Train, Y_Train, X_Validation, Y_Validation, epoch, learning_rate, weight_no, debug, PrintReport, patience):
        # Colors Order = GBYR
        train_accuracy = []
        y_axis_train = []
        x_axis = []
        v_accuracy = []
        y_axis_validation = []
        best_weight = []
        validate_accuracy = 0
        best_loss = float('inf')
        
        # Initializing weights for weight yellow, weights ranging from given weight number, with the size of the number of features
        weight_yellow = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        
        # Initializing weights for weight green, weights ranging from given weight number, with the size of the number of features
        weight_green = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        
        # Initializing weights for weight red, weights ranging from given weight number, with the size of the number of features
        weight_red = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        
        # Initializing weights for weight blue, weights ranging from given weight number, with the size of the number of features
        weight_blue = np.random.uniform(-weight_no, weight_no, len(X_Train[0]))
        
        print("               Training Model                            ")
        
        # Iterating through each epoch after updating weights for all data
        for e in range(epoch):
            
            # Taking all the data of encoding and its output into combined data
            combined_data = list(zip(X_Train, Y_Train))
            
            # Shuffling data at every epoch to train the model without overfitting
            random.shuffle(combined_data)
            
            # unzipping the value from the combined data into the encoding shuffled and target shuffled
            encoding_shuffled, target_shuffled = zip(*combined_data)
            
            # recoding the loss for the each epoch
            each_epoch_loss = []
            # recording the correct prediction for the each epoch
            correct_prediction_count = 0

            # Iterating through shuffled data from the combined data points
            for x_vector, y_vector in zip(encoding_shuffled, target_shuffled):
            
                # get the prediction or activattion function for the softmax regression to calculate the prediction for each class
                prediction_list = self.softmax_function(weight_blue, weight_yellow, weight_green, weight_red, x_vector)
                
                # Checking for the prediction, if its correct then increment else 0
                correct_prediction_count += 1 if np.argmax(prediction_list) == np.argmax(y_vector) else 0
                
                # calculating the loss for the each class with the y vector and prdiction values obtained from the activation function
                loss = self.cross_entropy_loss(prediction_list, y_vector)
                
                # appeding the loss for the data to calculate the loss over the epoch
                each_epoch_loss.append(loss)
                
                # update the weights using the stochastic gradient descent for the each class 
                new_weight = self.stochastic_gradient_descent(weight_blue, weight_yellow, weight_green, weight_red, x_vector, learning_rate, prediction_list, y_vector)

                weight_green = new_weight[0] 
                weight_blue = new_weight[1]
                weight_yellow = new_weight[2]
                weight_red  = new_weight[3]
            
            # Calcuating the average train acciracy for the each epoch
            train_accuracy_epoch = correct_prediction_count / len(Y_Train)
            train_accuracy.append(correct_prediction_count / len(Y_Train))
            
            x_axis.append(e)
            # Appending number of average loss for each epoch. 
            y_axis_train.append( sum(each_epoch_loss) / len(each_epoch_loss))
            
            # Calculating the validation accuracy for the each epoch with the updated weigts
            loss_validation, accuracy_validation = self.testing_model(X_Test=X_Validation, Y_Test=Y_Validation, weight=new_weight)
            v_accuracy.append(accuracy_validation)
            y_axis_validation.append(loss_validation)
            
            # Using the loss of the validation to not overfit the model
            if loss_validation < best_loss:
                best_loss = loss_validation
                validate_accuracy = accuracy_validation
                best_weight = new_weight
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print("Early stopping triggered.")
                    break
        
            if debug:         
                print(f"Epoch: {e:<3} Loss: {round(sum(each_epoch_loss) / len(each_epoch_loss), 5):<10} "
          f"Training Accuracy: {round(train_accuracy_epoch - 1e-5, 5):<10}")
        
        if PrintReport:        
            print("***********************************************************")
            print("    SoftMax REGRESSION MODEL TRAINING REPORT          ")
            print("***********************************************************")
            print("Accuracy for Training data Set   : ", round(train_accuracy_epoch- 1e-5, 5))
            print("Accuracy for Validation data Set : ", round(validate_accuracy,5))
            print("***********************************************************")
            print()
            
        best_weight = [weight_green, weight_blue, weight_yellow, weight_red]
        return best_weight, x_axis , y_axis_train , train_accuracy, y_axis_validation , v_accuracy,  train_accuracy_epoch, validate_accuracy
    
    
    # To test Model takes the parameters for the input spce and output space with the trained weights for the each class
    def testing_model(self, X_Test, Y_Test, weight):
        total_loss = 0
        correct_prediction_count = 0
        weight_green = weight[0] 
        weight_blue = weight[1]
        weight_yellow = weight[2]
        weight_red  = weight[3]
        
        # Iterating to all the input spce and output space
        for x_vector, y_vector in zip(X_Test, Y_Test):
            # Calculating the prediction using the updated weights
            prediction_list = self.softmax_function(weight_blue, weight_yellow, weight_green, weight_red, x_vector)
            # Calculating the correct prediction sum
            correct_prediction_count += np.argmax(prediction_list) == np.argmax(y_vector) 
            # Calcualting the loss on the predicted values 
            total_loss += self.cross_entropy_loss(prediction_list, y_vector)

        avg_loss = total_loss / len(Y_Test)
        avg_accuracy = correct_prediction_count / len(Y_Test)
        
        return avg_loss, avg_accuracy
   
    # Taking the dot product for the weight and x vector for the each data point
    def cal_z(self, weight, x_vector):
        return np.dot(weight, x_vector)

    # Using the Softmax function / Activation function to calculate the prediction bsaed on the weights parameters for each class
    def softmax_function(self, WB, WY, WG, WR, x_vector):
        # Calculating logits for each class
        z_red = self.cal_z(WR, x_vector)
        z_green = self.cal_z(WG, x_vector)
        z_blue = self.cal_z(WB, x_vector)
        z_yellow = self.cal_z(WY, x_vector)

        # Calculate softmax probabilities
        exp_red = np.exp(z_red)
        exp_green = np.exp(z_green)
        exp_blue = np.exp(z_blue)
        exp_yellow = np.exp(z_yellow)

        # Calculating the total proabilities 
        Total_sum = exp_red + exp_blue + exp_green + exp_yellow

        # Normalizing the probaility with dividing each class probabilities with the calculated sum
        function_green = exp_green / Total_sum
        function_blue = exp_blue / Total_sum
        function_yellow = exp_yellow / Total_sum
        function_red = exp_red / Total_sum

        return [function_green, function_blue, function_yellow, function_red]

    # Calculating loss with the given paramters from the softmax regression (Activation function) and the actual target values of the input space
    def cross_entropy_loss(self, predicted_values, target_values):
        # Sum of all the target values with the product of the log of predicted values
        loss = -np.sum(target_values * np.log(predicted_values))
        return loss

    
    def stochastic_gradient_descent(self, weight_blue, weight_yellow, weight_green, weight_red, x_vector, learning_rate, predicted_values, target):
        # GBYR

        # Calculating the error using the predicted values got from the activation fucntion and target value
        error = np.array(predicted_values) - np.array(target)
        
        x_vector = np.array(x_vector)

        # Update weights for each class
        new_weight_green = weight_green - (learning_rate * x_vector * error[0])
        new_weight_blue = weight_blue - (learning_rate * x_vector * error[1])
        new_weight_yellow = weight_yellow - (learning_rate * x_vector * error[2])
        new_weight_red = weight_red - (learning_rate * x_vector * error[3])
        
        updated_weights = [new_weight_green, new_weight_blue, new_weight_yellow, new_weight_red]
        return updated_weights

    def plot_bar(self, training, validation, testing):
        # Create a bar plot with training, validation, and testing values
        categories = ['Training', 'Validation', 'Testing']
        values = [training, validation, testing]
        plt.bar(categories, values, color=['red', 'grey', 'pink'],width= 0.5)
        plt.xlabel('Phase')
        plt.ylabel('Accuracy')
        plt.title(' Close this Graph to Finish excuting Further \n\n\n Accuracy Benchmarks')
        for i, value in enumerate(values):
            plt.text(i, value, str(value), ha='center', va='bottom')
        plt.show()

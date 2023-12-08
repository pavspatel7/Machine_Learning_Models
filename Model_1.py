import numpy as np
import random

# weight = np.random.uniform(-0.5, 0.5, features)

def encoding(grid):
    color_mapping = {
                     
                     '⬜️': [ 0, 0, 0, 0 ],
                     '🟥': [ 0, 0, 0, 1 ],
                     '🟨': [ 0, 0, 1, 0 ],
                     '🟦': [ 0, 1, 0, 0 ],
                     '🟩': [ 1, 0, 0, 0 ]
                    
                    }
    result = []
    result.append(1)
    for row in grid:
        for cell in row:
            result.extend(color_mapping.get(cell, [0, 0, 0, 0, 0]))
    return result

def training_model(encoding, target, weight, epoch, alpha, validation_x, validation_y):

    x_axis = []
    y_axis = []
    bestWeight = []
    validateAccuracy = 0
    for e in range(epoch):
        combined_data = list(zip(encoding, target))
        random.shuffle(combined_data)
        encoded_x_values, target_y_values = zip(*combined_data) 
        each_epoch_loss = []
        TrueCount = 0
        avg = 0
        print("--------------------------------------------------------------")
        for x, t in zip(encoded_x_values, target_y_values):
            # x = add_noise(np.array(x), 0.01)
            prediction = sigmoid_function(weight, x)
            TrueCount += 1  if classified_values(prediction) == t else 0
            loss = cross_entropy_loss(t, prediction)
            each_epoch_loss.append(loss)
            weight = gradient_descent(x, t, prediction, alpha, weight)
            
        avg = TrueCount / len(target)
        x_axis.append(e)
        y_axis.append(sum(each_epoch_loss) / len(each_epoch_loss))
        print(f"Loss: {y_axis[e]} at time {e}  Accuracy: {avg}")

        val_x, val_y, accuracy = test_model(validation_x, validation_y, weight)
        if accuracy > validateAccuracy:
            validateAccuracy = accuracy
            bestWeight = weight
            print(bestWeight)
        # else:
        #     weight = weight - np.mean(np.array(val_y))
    return bestWeight


def classified_values(prediction):
    if prediction > 0.5:
        return 1
    else:
        return 0

def test_model(encoding, target, weight):
    new_y = []
    new_x = []
    i = 0
    TrueCount = 0

    for x, s in zip(encoding, target):
        prediction = sigmoid_function(weight, x)
        TrueCount += 1  if classified_values(prediction) == s else 0
        loss = cross_entropy_loss(s, prediction)
        new_y.append(loss)
        new_x.append(i)
        i+=1

    avg = TrueCount / len(target)
    print(f"Accuracy for Testing Set {avg}")
    return new_x, new_y, avg

def sigmoid_function(weight, x_vector):
    weight_sum = np.dot(weight, x_vector)
    predicted_value = 1 / (1 + np.exp((-weight_sum)))
    return predicted_value


def cross_entropy_loss(targeted_value, predicted_value):
    loss = (-1 * float(targeted_value) * np.log(predicted_value)) - (
                float((1 - targeted_value)) * np.log(1 - predicted_value))
    return loss


def gradient_descent(features, target, prediction, alpha, weight):
    new_weights = weight + alpha * np.dot(features, (target-prediction))
    return new_weights 

def add_noise(one_hot_vector, noise_level):
    # Generate random noise with the same shape as the input vector
    noise = np.random.uniform(low=-noise_level, high=noise_level, size=one_hot_vector.shape)
    # Add noise to the one-hot vector
    noisy_vector = one_hot_vector + noise
    # Clip values to ensure they remain within [0, 1]
    noisy_vector = np.clip(noisy_vector, 0, 1)
    # Ensure the vector remains normalized (sums to 1)
    noisy_vector /= np.sum(noisy_vector)
    return noisy_vector

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
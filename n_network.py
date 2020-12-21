# Jun Sung Tak
# ANN for iris classification
# 16/12/2020

import numpy as np
import pandas

num_fields = 4  #Number of numerical fields
num_t_data = 120  #Training data length
num_d_data = 30 #Deploy data length
f_name = "iris.txt"

# min, max values of each columns 
s_len_min = 4.3
s_len_max = 7.9
s_width_min = 2.0
s_width_max = 4.4
p_len_min = 1.0
p_len_max = 6.9
p_width_min = 0.1
p_width_max = 2.5

# Random seed for consistent randomizing
np.random.seed(1337)

# This was used to find the min and maxs for each columns
# This is function does not affect the program
def min_max(x):
    return pandas.Series(index=['min', 'max'], data=[x.min(), x.max()])

#For training
input_matrix = [[0 for i in range(num_fields)] for j in range(num_t_data)]
output_matrix = []
t_input_matrix = [[0 for i in range(num_fields)] for j in range(num_d_data)]
t_output_matrix = []


title = ["s_length", "s_width", "p_length", "p_width", "label"]
data = pandas.read_csv(f_name, names = title)  #Organize data into tables
# print(data.apply(min_max) # Displays the min and maxes of each column

# Shuffle data so that network gets a variety of samples
data_shuffled = data.sample(frac=1).reset_index(drop=True)

# Populate training set
for i in range(num_t_data):  #put elements in matrix form
    input_matrix[i][0] = data_shuffled['s_length'][i]
    input_matrix[i][1] = data_shuffled['s_width'][i]
    input_matrix[i][2] = data_shuffled['p_length'][i]
    input_matrix[i][3] = data_shuffled['p_width'][i]
for i in range(num_t_data):  #Corresponding answer to data sets
    if data_shuffled["label"][i] == "Iris-setosa":
        output_matrix.append([1, 0, 0])
    elif data_shuffled["label"][i] == "Iris-versicolor":
        output_matrix.append([0, 1, 0])
    else:
        output_matrix.append([0, 0, 1])

# Populate testing set
for x in range(num_d_data):
    t_input_matrix[x][0] = data_shuffled['s_length'][x + num_t_data]
    t_input_matrix[x][1] = data_shuffled['s_width'][x + num_t_data]
    t_input_matrix[x][2] = data_shuffled['p_length'][x + num_t_data]
    t_input_matrix[x][3] = data_shuffled['p_width'][x + num_t_data]
for x in range(num_d_data):
    if data_shuffled["label"][x + num_t_data] == "Iris-setosa":
        t_output_matrix.append([1, 0, 0])
    elif data_shuffled["label"][x + num_t_data] == "Iris-versicolor":
        t_output_matrix.append([0, 1, 0])
    else:
        t_output_matrix.append([0, 0, 1])

# Min-max scaling for both set
for i in range(num_t_data):
    input_matrix[i][0] = (input_matrix[i][0] - s_len_min) / (s_len_max - s_len_min)
    input_matrix[i][1] = (input_matrix[i][1] - s_width_min) / (s_width_max - s_width_min)
    input_matrix[i][2] = (input_matrix[i][2] - p_len_min) / (p_len_max - p_len_min)
    input_matrix[i][3] = (input_matrix[i][3] - p_width_min) / (p_width_max - p_width_min)
for i in range(num_d_data):
    t_input_matrix[i][0] = (t_input_matrix[i][0] - s_len_min) / (s_len_max - s_len_min)
    t_input_matrix[i][1] = (t_input_matrix[i][1] - s_width_min) / (s_width_max - s_width_min)
    t_input_matrix[i][2] = (t_input_matrix[i][2] - p_len_min) / (p_len_max - p_len_min)
    t_input_matrix[i][3] = (t_input_matrix[i][3] - p_width_min) / (p_width_max - p_width_min)


# Class representing a single layer of neurons
class NLayer():
    def __init__(self, num_neuron, num_input):
        self.weight = 2 * np.random.random((num_input, num_neuron)) - 1
        self.num_neuron = num_neuron
        self.num_input = num_input
    
    def get_neuron_num(self):
        return self.num_neuron
    
    def get_input_num(self):
        return self.num_input

# Class representing the entire network. Contains the layers
class NNetwork():
    def __init__(self, h_layer1, h_layer2):
        self.h_layer1 = h_layer1
        self.h_layer2 = h_layer2
    
    def sigmoid(self, val):
        return 1 / (1 + np.exp(-val))
    
    def d_sigmoid(self, val):
        return self.sigmoid(val) * (1 - self.sigmoid(val))
    
    # Feed forward
    def feed_forward(self, inp):
        layer1_out = self.sigmoid(np.dot(inp, self.h_layer1.weight)) #dot product of input and weights
        layer2_out = self.sigmoid(np.dot(layer1_out, self.h_layer2.weight))
        return layer1_out, layer2_out
    
    # Feeds forward, calculates error, and makes adjustments
    def back_prop_adj(self, t_input, t_output, epoch):
        for i in range(epoch):
            out = self.feed_forward(t_input)
            out1 = out[0]
            out2 = out[1]
            layer2_err = t_output.T - out2
            delta_layer2_err = layer2_err * self.d_sigmoid(out2) * 0.01
            layer1_err = delta_layer2_err.dot(self.h_layer2.weight.T)
            delta_layer1_err = layer1_err * self.d_sigmoid(out1) * 0.01
            layer1_adj = t_input.T.dot(delta_layer1_err)
            layer2_adj = out1.T.dot(delta_layer2_err)
            self.h_layer1.weight += layer1_adj
            self.h_layer2.weight += layer2_adj
    
    # Display weights
    def show_w(self):
        l1_neuron = self.h_layer1.get_neuron_num
        l1_input = self.h_layer1.get_input_num
        l2_neuron = self.h_layer2.get_neuron_num
        l2_input = self.h_layer2.get_input_num
        print("****||||Layer1: 5 neurons with 4 inputs||||****")
        print(self.h_layer1.weight)
        print("****||||Layer2: 3 neurons with 5 inputs||||****")
        print(self.h_layer2.weight)
        print()

# Undo scaling for display
def undo_minmax(arr):
    aux = [0, 0, 0, 0]
    aux[0] = round(arr[0] * (s_len_max - s_len_min) + s_len_min, 1)
    aux[1] = round(arr[1] * (s_width_max - s_width_min) + s_width_min, 1)
    aux[2] = round(arr[2] * (p_len_max - p_len_min) + p_len_min, 1)
    aux[3] = round(arr[3] * (p_width_max - p_width_min) + p_width_min, 1)
    return aux

# Decoding for determing results
def decode_result(arr):
    if arr[0] == 1 and arr[1] == 0 and arr[2] == 0:
        return 1
    if arr[0] == 0 and arr[1] == 1 and arr[2] == 0:
        return 2
    if arr[0] == 0 and arr[1] == 0 and arr[2] == 1:
        return 3

# Classifies 1 flower
def pred_classify(inp_arr, out_arr, nnetwork):
    ans = None
    h_state, out = nnetwork.feed_forward(inp_arr)
    print("Prediction: ", end="")
    print(out)
    print("Class: ", end="")
    if out[0] > 0 and out[0] > out[1] and out[0] > out[2]:
        print("Iris-setosa")
        ans = [1, 0, 0]
    elif out[1] > 0 and out[1] > out[0] and out[1] > out[2]:
        print("Iris-versicolor")
        ans = [0, 1, 0]
    else:
        print("Iris-verginica")
        ans = [0, 0, 1]
    if decode_result(ans) == decode_result(out_arr):
        print("Accurate Prediction!\n")
        return True
    print("Failed...\n")
    return False

# Testing phase loop
def calculate_accuracy(inp, out, nnetwork):
    counter = 0
    for i in range(len(inp)):
        print("Test: " + str(i + 1))
        input_data = inp[i]
        target = out[i]
        print("Input: ", end="")
        print(undo_minmax(input_data))
        print("Target: ", end="")
        print(target)
        ans = pred_classify(input_data, target, nnetwork)
        if ans == True:
            counter += 1
    accuracy = float(counter) / float(len(inp))
    return accuracy

def classify(inp, nnetwork):
    print("Input : ", end="")
    print(inp)
    inp[0] = (inp[0] - s_len_min) / (s_len_max - s_len_min)
    inp[1] = (inp[1] - s_width_min) / (s_width_max - s_width_min)
    inp[2] = (inp[2] - p_len_min) / (p_len_max - p_len_min)
    inp[3] = (inp[3] - p_width_min) / (p_width_max - p_width_min)
    h_state, out = nnetwork.feed_forward(inp)
    print("Output: ", end="")
    print(out)
    print("Class: ", end="")
    if out[0] > 0 and out[0] > out[1] and out[0] > out[2]:
        print("Iris-setosa")
    elif out[1] > 0 and out[1] > out[0] and out[1] > out[2]:
        print("Iris-versicolor")
    else:
        print("Iris-verginica")




# Initialize 2 layers
a = NLayer(5, 4) # 4 input 5 neuron
b = NLayer(3, 5) # 5 input 3 neuron
# Initialize network with the layers
network = NNetwork(a, b)

training_set_input = np.array(input_matrix)
training_set_output = np.array(output_matrix).T # Transpose for calculation

print("\tPhase 1: Initial randomized weights before training...")
network.show_w()
network.back_prop_adj(training_set_input, training_set_output, 15000) # epoch = 15000

print("\tPhase 2: New weights after training phase...")
network.show_w()
print("Training complete!\n")

print("\tPhase 3: Testing accuracy of network...")
g = calculate_accuracy(t_input_matrix, t_output_matrix, network)
print("Testing complete!\n")

print("Testing set prediction accuracy: ", end="")
print(g * 100, end="")
print("%\n")





# TO MANUALLY ENTER DATA MODIFY THE manual_input VARIABLE
#               s_l  s_w  p_l  p_w
manual_input = [4.0, 2.3, 5.3, 2.0]
classify(manual_input, network)


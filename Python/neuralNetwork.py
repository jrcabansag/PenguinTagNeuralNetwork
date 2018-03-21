import sys
import tensorflow as tf
import numpy as np
import json

#Starter code copied from CS230 jupyter notebook

def open_data(file):
    data_file = open(file, 'r')
    data = []
    for line in data_file:
        data_array = [int(x) for x in line.strip().split("|")]
        data.append(data_array)
    data = np.array(data).T
    return data[2:], data[1]

def one_hot_Y(Y, classes):
    C = tf.constant(classes, name="C")
    one_hot_matrix = tf.one_hot(Y, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot
  
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X_input")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y_input")

    return X, Y

def initialize_parameters(n_x, n_y):
    tf.set_random_seed(1)

    '''
    SHALLOW RANDOM INIT
    '''
    # W1 = tf.get_variable("W1", [50,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b1 = tf.get_variable("b1", [50,1], initializer = tf.zeros_initializer())
    # W2 = tf.get_variable("W2", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b2 = tf.get_variable("b2", [50,1], initializer = tf.zeros_initializer())
    # W3 = tf.get_variable("W3", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b3 = tf.get_variable("b3", [50,1], initializer = tf.zeros_initializer())
    # W4 = tf.get_variable("W4", [n_y,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b4 = tf.get_variable("b4", [n_y,1], initializer = tf.zeros_initializer())
    # parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4":b4}

    '''
    SHALLOW PRESET INIT
    '''
    # parameters_dict = json.loads(open("../Models/SmartBalanced/SmartBalanced2600Iterations.json").read())
    # W1 = tf.get_variable("W1", initializer = tf.constant(parameters_dict["W1"]))
    # b1 = tf.get_variable("b1", initializer = tf.constant(parameters_dict["b1"]))
    # W2 = tf.get_variable("W2", initializer = tf.constant(parameters_dict["W2"]))
    # b2 = tf.get_variable("b2", initializer = tf.constant(parameters_dict["b2"]))
    # W3 = tf.get_variable("W3", initializer = tf.constant(parameters_dict["W3"]))
    # b3 = tf.get_variable("b3", initializer = tf.constant(parameters_dict["b3"]))
    # W4 = tf.get_variable("W4", initializer = tf.constant(parameters_dict["W4"]))
    # b4 = tf.get_variable("b4", initializer = tf.constant(parameters_dict["b4"]))
    # parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4":b4}

    '''
    DEEP BATCH NORM RANDOM INIT
    '''
    # W1 = tf.get_variable("W1", [10,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # W2 = tf.get_variable("W2", [10,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # W3 = tf.get_variable("W3", [10,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # W4 = tf.get_variable("W4", [10,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # W5 = tf.get_variable("W5", [10,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # W6 = tf.get_variable("W6", [10,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # W7 = tf.get_variable("W7", [10,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # W8 = tf.get_variable("W8", [n_y,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5, "W6": W6, "W7": W7, "W8": W8}

    '''
    DEEP BATCH NORM PRESET INIT
    '''
    # parameters_dict = json.loads(open("../Models/NaturalDeepBalanced/NaturalDeepBalanced2600Iterations.json").read())
    # W1 = tf.get_variable("W1", initializer = tf.constant(parameters_dict["W1"]))
    # W2 = tf.get_variable("W2", initializer = tf.constant(parameters_dict["W2"]))
    # W3 = tf.get_variable("W3", initializer = tf.constant(parameters_dict["W3"]))
    # W4 = tf.get_variable("W4", initializer = tf.constant(parameters_dict["W4"]))
    # W5 = tf.get_variable("W5", initializer = tf.constant(parameters_dict["W5"]))
    # W6 = tf.get_variable("W6", initializer = tf.constant(parameters_dict["W6"]))
    # W7 = tf.get_variable("W7", initializer = tf.constant(parameters_dict["W7"]))
    # W8 = tf.get_variable("W8", initializer = tf.constant(parameters_dict["W8"]))
    # parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5, "W6": W6, "W7": W7, "W8": W8}

    '''
    DEEP RANDOM INIT
    '''
    # W1 = tf.get_variable("W1", [50,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b1 = tf.get_variable("b1", [50,1], initializer = tf.zeros_initializer())
    # W2 = tf.get_variable("W2", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b2 = tf.get_variable("b2", [50,1], initializer = tf.zeros_initializer())
    # W3 = tf.get_variable("W3", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b3 = tf.get_variable("b3", [50,1], initializer = tf.zeros_initializer())
    # W4 = tf.get_variable("W4", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b4 = tf.get_variable("b4", [50,1], initializer = tf.zeros_initializer())
    # W5 = tf.get_variable("W5", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b5 = tf.get_variable("b5", [50,1], initializer = tf.zeros_initializer())
    # W6 = tf.get_variable("W6", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b6 = tf.get_variable("b6", [50,1], initializer = tf.zeros_initializer())
    # W7 = tf.get_variable("W7", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b7 = tf.get_variable("b7", [50,1], initializer = tf.zeros_initializer())
    # W8 = tf.get_variable("W8", [n_y,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # b8 = tf.get_variable("b8", [n_y,1], initializer = tf.zeros_initializer())
    # parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4":b4, "W5": W5, "b5": b5, "W6": W6, "b6": b6, "W7": W7, "b7": b7, "W8": W8, "b8":b8}

    '''
    DEEP PRESET INIT
    '''
    parameters_dict = json.loads(open("../Models/DeepNoNorm/DeepNoNorm4300Iterations.json").read())
    W1 = tf.get_variable("W1", initializer = tf.constant(parameters_dict["W1"]))
    b1 = tf.get_variable("b1", initializer = tf.constant(parameters_dict["b1"]))
    W2 = tf.get_variable("W2", initializer = tf.constant(parameters_dict["W2"]))
    b2 = tf.get_variable("b2", initializer = tf.constant(parameters_dict["b2"]))
    W3 = tf.get_variable("W3", initializer = tf.constant(parameters_dict["W3"]))
    b3 = tf.get_variable("b3", initializer = tf.constant(parameters_dict["b3"]))
    W4 = tf.get_variable("W4", initializer = tf.constant(parameters_dict["W4"]))
    b4 = tf.get_variable("b4", initializer = tf.constant(parameters_dict["b4"]))
    W5 = tf.get_variable("W5", initializer = tf.constant(parameters_dict["W5"]))
    b5 = tf.get_variable("b5", initializer = tf.constant(parameters_dict["b5"]))
    W6 = tf.get_variable("W6", initializer = tf.constant(parameters_dict["W6"]))
    b6 = tf.get_variable("b6", initializer = tf.constant(parameters_dict["b6"]))
    W7 = tf.get_variable("W7", initializer = tf.constant(parameters_dict["W7"]))
    b7 = tf.get_variable("b7", initializer = tf.constant(parameters_dict["b7"]))
    W8 = tf.get_variable("W8", initializer = tf.constant(parameters_dict["W8"]))
    b8 = tf.get_variable("b8", initializer = tf.constant(parameters_dict["b8"]))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4":b4, "W5": W5, "b5": b5, "W6": W6, "b6": b6, "W7": W7, "b7": b7, "W8": W8, "b8":b8}


    return parameters

def forward_propagation(X, parameters):

    '''
    SHALLOW NETWORK
    '''

    # W1 = parameters['W1']
    # b1 = parameters['b1']
    # W2 = parameters['W2']
    # b2 = parameters['b2']
    # W3 = parameters['W3']
    # b3 = parameters['b3']
    # W4 = parameters['W4']
    # b4 = parameters['b4']
    # Z1 = tf.add(tf.matmul(W1,X), b1)
    # A1 = tf.nn.relu(Z1)
    # Z2 = tf.add(tf.matmul(W2,A1), b2)     
    # A2 = tf.nn.relu(Z2)
    # Z3 = tf.add(tf.matmul(W3,A2), b3)
    # A3 = tf.nn.relu(Z3)
    # Z4 = tf.add(tf.matmul(W4,A3), b4)
    # return Z4, parameters


    '''
    DEEP NETWORK
    '''

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    W7 = parameters['W7']
    b7 = parameters['b7']
    W8 = parameters['W8']
    b8 = parameters['b8']
    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)     
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4,A3), b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5,A4), b5)     
    A5 = tf.nn.relu(Z5)
    Z6 = tf.add(tf.matmul(W6,A5), b6)
    A6 = tf.nn.relu(Z6)
    Z7 = tf.add(tf.matmul(W7,A6), b7)
    A7 = tf.nn.relu(Z7)
    Z8 = tf.add(tf.matmul(W8,A7), b8)
    return Z8, parameters


    '''
    DEEP NETWORK W/ BATCH NORM
    '''

    # W1 = parameters['W1']
    # W2 = parameters['W2']
    # W3 = parameters['W3']
    # W4 = parameters['W4']
    # W5 = parameters['W5']
    # W6 = parameters['W6']
    # W7 = parameters['W7']
    # W8 = parameters['W8']
    # Z1 = tf.matmul(W1,X)
    # Z1_mean, Z1_var = tf.nn.moments(Z1,[1], keep_dims=True)
    # Z1_norm = tf.nn.batch_normalization(Z1, Z1_mean, Z1_var, None, None, 0)   
    # A1 = tf.nn.relu(Z1_norm)
    # Z2 = tf.matmul(W2,A1)  
    # Z2_mean, Z2_var = tf.nn.moments(Z2,[1], keep_dims=True)
    # Z2_norm = tf.nn.batch_normalization(Z2, Z2_mean, Z2_var, None, None, 0)   
    # A2 = tf.nn.relu(Z2_norm)
    # Z3 = tf.matmul(W3,A2)
    # Z3_mean, Z3_var = tf.nn.moments(Z3,[1], keep_dims=True)
    # Z3_norm = tf.nn.batch_normalization(Z3, Z3_mean, Z3_var, None, None, 0)   
    # A3 = tf.nn.relu(Z3_norm)
    # Z4 = tf.matmul(W4,A3)
    # Z4_mean, Z4_var = tf.nn.moments(Z4,[1], keep_dims=True)
    # Z4_norm = tf.nn.batch_normalization(Z4, Z4_mean, Z4_var, None, None, 0)   
    # A4 = tf.nn.relu(Z4_norm)
    # Z5 = tf.matmul(W5,A4)    
    # Z5_mean, Z5_var = tf.nn.moments(Z5,[1], keep_dims=True)
    # Z5_norm = tf.nn.batch_normalization(Z5, Z5_mean, Z5_var, None, None, 0)   
    # A5 = tf.nn.relu(Z5_norm)
    # Z6 = tf.matmul(W6,A5)
    # Z6_mean, Z6_var = tf.nn.moments(Z6,[1], keep_dims=True)
    # Z6_norm = tf.nn.batch_normalization(Z6, Z6_mean, Z6_var, None, None, 0)   
    # A6 = tf.nn.relu(Z6_norm)
    # Z7 = tf.matmul(W7,A6)
    # Z7_mean, Z7_var = tf.nn.moments(Z7,[1], keep_dims=True)
    # Z7_norm = tf.nn.batch_normalization(Z7, Z7_mean, Z7_var, None, None, 0)   
    # A7 = tf.nn.relu(Z7_norm)
    # Z8 = tf.matmul(W8,A7)

    # parameters["Z1_mean"] = Z1_mean
    # parameters["Z1_var"] = Z1_var
    # parameters["Z2_mean"] = Z2_mean
    # parameters["Z2_var"] = Z2_var
    # parameters["Z3_mean"] = Z3_mean
    # parameters["Z3_var"] = Z3_var
    # parameters["Z4_mean"] = Z4_mean
    # parameters["Z4_var"] = Z4_var
    # parameters["Z5_mean"] = Z5_mean
    # parameters["Z5_var"] = Z5_var
    # parameters["Z6_mean"] = Z6_mean
    # parameters["Z6_var"] = Z6_var
    # parameters["Z7_mean"] = Z7_mean
    # parameters["Z7_var"] = Z7_var
    # return Z8, parameters

def compute_cost(Z, Y):
    # logits = tf.transpose(Z)
    # labels = tf.transpose(Y)
    # labels = tf.cast(labels, tf.int32)
    # # logits = Z
    # # labels = Y
    # # print(logits)
    # # print(labels)
    # class_weights = tf.constant([0.1, 0.2, 0.15, 0.2, 0.15, 0.2])
    # weights = tf.gather(class_weights, labels)
    # #cost_array = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    # #print(cost_array)
    # #cost = tf.reduce_mean(cost_array)
    # cost_array = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights).eval()
    # cost = tf.reduce_mean(cost_array)
    # return cost, cost_array

    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    class_weights = tf.constant([[274744/158840, 274744/14720, 274744/24738, 274744/14720, 274744/24738, 274744/36988]])
    weights = tf.reduce_sum(class_weights * labels, axis=1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    weighted_losses = unweighted_losses * weights
    cost = tf.reduce_mean(weighted_losses)
    cost_array = weighted_losses

    return cost, cost_array


def split_into_moves(X_data, Y_data):
    split_dict = {}
    for x in range(6):
        split_dict[x] = []
        data_coords = np.where(Y_data == x)
        for m in data_coords[0]:
            split_dict[x].append(X_data[:,m])
        split_dict[x] = np.array(split_dict[x]).T
        print(split_dict[x].shape)
    return split_dict

def main():
    tf.set_random_seed(1)

    #X_train, Y_train = open_data("../Data/Week10ParsedAugmentedDirectionsOnly.txt")
    X_train, Y_train = open_data("../Data/Week10ParsedAugmented.txt")
    train_dict = split_into_moves(X_train, Y_train)

    Y_train = one_hot_Y(Y_train, 6)

    X_test, Y_test = open_data("../Data/finalTestDataParsed.txt")
    test_dict = split_into_moves(X_test, Y_test)

    Y_test = one_hot_Y(Y_test, 6)

    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]
    m = X_train.shape[1]

    X, Y = create_placeholders(n_x, n_y)
    #X_mean, X_var = tf.nn.moments(X,[0])
    #X_norm = tf.nn.batch_normalization(X, X_mean, X_var, None, None, 0)
    parameters = initialize_parameters(n_x, n_y)
    #Z = forward_propagation(X_norm, parameters)
    Z, parameters = forward_propagation(X, parameters)
    print(parameters)
    A = Z
    cost, cost_array = compute_cost(A, Y)
    #optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate = 0.05).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005).minimize(cost)
    my_learning_rate = 0.0000001
    optimizer = tf.train.AdamOptimizer(learning_rate = my_learning_rate).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        preset_int = 4300
        min_cost = 10000000
        for run in range(100001):
            _, my_cost, my_cost_array, final_params = sess.run([optimizer, cost, cost_array, parameters], feed_dict={X: X_train, Y: Y_train})
            if my_cost > min_cost:
                print("COST BECAME LARGER BY {} IN RUN {}".format(my_cost-min_cost, run+preset_int))
            else:
                print("RUN {} IS GOOD! MIN COST DECREASED BY {}".format(run+preset_int, min_cost-my_cost))
                min_cost = my_cost
            if run % 100 == 0:
                print("Cost at {} is {}".format(run+preset_int, my_cost))
                print("Cost array at {} is {}".format(run+preset_int, my_cost_array))
                correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
                print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
                for x in range(6):
                    curr_one_hot_Y = one_hot_Y(np.array([x for a in range(train_dict[x].shape[1])]), 6)
                    print(curr_one_hot_Y.shape)
                    print("Train accuracy for move {} is {}".format(x, accuracy.eval({X: train_dict[x], Y: curr_one_hot_Y})))
                    curr_test_one_hot_Y = one_hot_Y(np.array([x for a in range(test_dict[x].shape[1])]), 6)
                    print(curr_test_one_hot_Y.shape)
                    print("Test accuracy for move {} is {}".format(x, accuracy.eval({X: test_dict[x], Y: curr_test_one_hot_Y})))
                for key in final_params:
                    final_params[key] = final_params[key].tolist();
                with open("../Models/DeepNoNorm/DeepNoNorm"+str(run+preset_int)+"Iterations.json", 'w') as outfile:
                    json.dump(final_params, outfile)

main()
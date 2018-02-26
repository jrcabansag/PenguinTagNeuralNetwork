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

    W1 = tf.get_variable("W1", [50,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [50,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [50,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [50,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [n_y,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [n_y,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4":b4}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    Z1 = tf.add(tf.matmul(W1,X), b1)  
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)  
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)  
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4,A3), b4)  

    return Z4

def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

def main():
    tf.set_random_seed(1)

    X_train, Y_train = open_data("../Data/kingrickyParsedAugmented.txt")

    Y_train = one_hot_Y(Y_train, 6)

    X_test, Y_test = open_data("../Data/kingParsed.txt")

    Y_test = one_hot_Y(Y_test, 6)

    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]
    m = X_train.shape[1]

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n_x, n_y)
    Z = forward_propagation(X, parameters)
    A = Z
    cost = compute_cost(A, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for run in range(10001):
            _, my_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            if run % 100 == 0:
                print("Cost at {} is {}".format(run, my_cost))
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        final_params = sess.run(parameters)

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        for key in final_params:
            final_params[key] = final_params[key].tolist();
        output_file = open("../Models/model1.json", 'w')
        output_file.write(json.dumps(final_params));

main()
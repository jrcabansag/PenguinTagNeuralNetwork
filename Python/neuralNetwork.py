import sys
import tensorflow as tf
import numpy as np
import json

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

    W1 = tf.get_variable("W1", [n_y,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [n_y,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    Z1 = tf.add(tf.matmul(W1,X), b1)  

    return Z1

def sigmoid(Z):
    return tf.nn.sigmoid(Z);

def relu(Z):
    return tf.nn.relu(Z);

def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

def main():
    # hello = tf.constant('Hello, TensorFlow!')
    # sess = tf.Session()
    # print(sess.run(hello))
    tf.set_random_seed(1)

    X_train, Y_train = open_data("../Data/trainData1ParsedAugmentedDirectionsOnly.txt")

    Y_train = one_hot_Y(Y_train, 6)

    X_test, Y_test = open_data("../Data/testData1ParsedDirectionsOnly.txt")

    Y_test = one_hot_Y(Y_test, 6)

    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]
    m = X_train.shape[1]

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n_x, n_y)
    Z = forward_propagation(X, parameters)
    A = sigmoid(Z)
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
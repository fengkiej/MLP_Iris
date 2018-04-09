from math import exp
from numpy import random, dot, append, transpose
from statistics import mean


class Model:
    # layers = [number_of_input_nodes, number_of_hidden_nodes, ... , number_of_output_nodes]
    # learning_rate = float
    # tau = [tau hidden layer 1, tau hidden layer 2, ..., tau hidden layer n, tau output layer]
    def __init__(self, learning_rate, layers):
        if layers is None:
            raise ValueError

        self.learning_rate = learning_rate
        self.layers = layers
        self.layer_weights = []
        self.error = []
        self.accuracy = []

        for i in range(0, len(layers) - 1):
            self.layer_weights.append(random.rand(layers[i + 1], layers[i] + 1))

    def train(self, training_dataset, validation_dataset, epoch):
        random.shuffle(training_dataset)
        # append for bias
        for data in training_dataset:
            data.insert(len(data) - 1, 1)

        for i in range(0, epoch):
            err = []
            # print(self.layer_weights)
            for data in training_dataset:
                fact = []
                for i in range(0, self.layers[-1]):
                    fact.append(0)
                fact[int(data[-1])] = 1

                z = self.feedforward(data[:-1])
                prediction = z[-1]
                err.append(self.get_error(prediction, fact))

                self.backpropagation(prediction, fact, z)

            self.validate_model(validation_dataset)
            self.error.append(mean(err))

        print(self.error)

        for data in training_dataset:
            data.pop(-2)

    def validate_model(self, validation_dataset):
        predict_true = 0

        for data in validation_dataset:
            data.insert(len(data) - 1, 1)
            z = self.feedforward(data[:-1])
            prediction = z[-1][:-1]
            maxindex = 0
            maxval = 0
            for i in range(0, len(prediction)):
                if prediction[i] > maxval:
                    maxval = prediction[i]
                    maxindex = i
            #print(data, prediction, maxindex)
            if maxindex == data[-1]:
                predict_true = predict_true + 1

        print("right predictions: ", predict_true)
        self.accuracy.append(predict_true / len(validation_dataset))
        for data in validation_dataset:
            data.pop(-2)

    def backpropagation(self, prediction, fact, z):
        tau = []
        t = []
        for i in range(0, self.layers[-1]):
            t.append(self.get_init_tau(prediction[i], fact[i]))

        tau.insert(0, t)

        for i in range(len(self.layers) - 2, 0, -1):
            weights = transpose(transpose(self.layer_weights[i])[:-1])
            temp_tau = dot(tau[0], weights)
            for j in range(0, self.layers[i]):
                temp_tau = temp_tau * (1 - z[i][j]) * z[i][j]
            tau.insert(0, temp_tau)
        self.update_weights(tau, z)

    def feedforward(self, data):
        # print(data)
        z = [data]
        for i in range(0, len(self.layers) - 1):
            evaluation = self.predict(self.layer_weights[i], z[i])
            evaluation = append(evaluation, 1)
            z.append(evaluation)

        return z

    def update_weights(self, tau, z):
        #print(self.layer_weights)
        #print(tau)
        #print(z)
        for i in range(0, len(self.layer_weights)):
            for j in range(0, len(self.layer_weights[i])):
                for k in range(0, len(self.layer_weights[i][j])):
                    self.layer_weights[i][j][k] = self.layer_weights[i][j][k] - self.learning_rate * tau[i][j] * z[i][k]
        #print(self.layer_weights)
        #print("####")

    @staticmethod
    def predict(weights, data):
        return list(map(Model.sigmoid, Model.h(weights, data)))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def h(weights, inputs):
        return dot(weights, inputs)

    @staticmethod
    def get_init_tau(prediction, fact):
        return 2 * (prediction - fact) * (1 - prediction) * prediction

    @staticmethod
    def get_error(prediction, fact):
        error = 0
        for i in range(0, len(fact)):
            error = error + (prediction[i] - fact[i]) ** 2
        return error/len(fact)

    @staticmethod
    def get_delta_weights(h, tau):
        return h * tau

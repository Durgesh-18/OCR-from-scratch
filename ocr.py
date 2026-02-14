import numpy as np
import json
import math
import os

class OCRNeuralNetwork:
    NN_FILE_PATH = "nn.json"
    LEARNING_RATE = 0.1

    def __init__(self, num_hidden_nodes=25, use_file=True):
        self._use_file = use_file

        self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)

        self.input_layer_bias = np.zeros((num_hidden_nodes, 1))
        self.hidden_layer_bias = np.zeros((10, 1))

        if use_file and os.path.exists(self.NN_FILE_PATH):
            self._load()

    def _rand_initialize_weights(self, size_in, size_out):
        return np.random.rand(size_out, size_in) * 0.12 - 0.06

    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.exp(-z))

    def sigmoid(self, z):
        return np.vectorize(self._sigmoid_scalar)(z)

    def sigmoid_prime(self, z):
        s = self.sigmoid(z)
        return np.multiply(s, (1 - s))

    def train(self, training_data):
        for data in training_data:
            y1 = np.dot(self.theta1, np.array(data["y0"]).reshape(-1, 1))
            sum1 = y1 + self.input_layer_bias
            y1 = self.sigmoid(sum1)

            y2 = np.dot(self.theta2, y1)
            y2 = y2 + self.hidden_layer_bias
            y2 = self.sigmoid(y2)

            actual = np.zeros((10, 1))
            actual[data["label"]] = 1

            output_errors = actual - y2
            hidden_errors = np.multiply(
                np.dot(self.theta2.T, output_errors),
                self.sigmoid_prime(sum1)
            )

            self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, np.array(data["y0"]).reshape(1, -1))
            self.theta2 += self.LEARNING_RATE * np.dot(output_errors, y1.T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(self.theta1, np.array(test).reshape(-1, 1))
        y1 = y1 + self.input_layer_bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(self.theta2, y1)
        y2 = y2 + self.hidden_layer_bias
        y2 = self.sigmoid(y2)

        return int(np.argmax(y2))

    def save(self):
        if not self._use_file:
            return

        nn_data = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }

        with open(self.NN_FILE_PATH, "w") as f:
            json.dump(nn_data, f)

    def _load(self):
        with open(self.NN_FILE_PATH) as f:
            nn = json.load(f)

        self.theta1 = np.array(nn["theta1"])
        self.theta2 = np.array(nn["theta2"])
        self.input_layer_bias = np.array(nn["b1"])
        self.hidden_layer_bias = np.array(nn["b2"])

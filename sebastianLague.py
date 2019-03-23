# Based on: https://www.youtube.com/watch?v=8bNIkfRJZpo&t=388s
import numpy as np

class neuralNetwork:
    def __init__(self, layerSizes):
        """
        Create a neural network from an array of layer sizes.
        First layer is input, last layer is output. Between are hidden layers.
        Initializes weights and biases to random numbers.
        """

        # The weights and biases between each layer can be modeled as a matrix.
        # The number of nodes in the layer on the left determines the number of
        # columns in the matrix. Vice-versa, the right side determines rows.
        weightShapes = [(layerSizes[i], layerSizes[i-1]) for i in range(1,len(layerSizes))]

        # Generate the weight matrices with the above sizes and set each weight randomly.
        # If the weights are all the same number, the network changes each weight the
        # same amount, so it never learns anything.
        self.weights = [np.random.standard_normal(i) for i in weightShapes]

        # Biases are 1 column wide. Input layer doesn't have biases. These can start as 0
        self.biases = [np.zeroes((i, 1)) for i in layerSizes[1:]]

    def feedForward(self, a):
        # Step through each set of weights and bias together
        for (w, b) in zip(self.weights, self.biases):
            # Matrix multiply weight * activation of previous layer + bias
            # and then squish it inside a sigmoid function
            a = self.sigmoid(np.matmul(w, a) + b)
        return(a)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function. Large positive inputs return close to 1. Large negative
        inputs return close to 0.
        """
        return(1/(1+np.exp(-x)))


#layerSizes = [2, 3, 5, 2]



for i in weights:
    print(i)

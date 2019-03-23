# Video series this is based on: https://www.youtube.com/watch?v=aircAruvnKk

from random import random
import pickle
import math
import numpy # Dependency: pip install numpy

def sigmoid(x):
	# Sigmoid "Squishification" function
	return 1 / (1 + math.exp(-x))

class neuron:
	# Each neuron has a set of weights for each input, a bias for itself, and a value between 0 and 1
	def __init__(self, numWeights):
		self.weights = [random() for i in range(0,numWeights)] # Initialize to random numbers between 0 and 1
		self.bias = random()
		self.value = 0.0

	# Get the new value of this neuron given the values of its inputs
	def update(self, inValues):
		
		# Make sure the input values are the correct type
		if isinstance(inValues[0], neuron):
			inValues = [i.value for i in inValues]
		if type(inValues[0]) == int:
			inValues = [float(i) for i in inValues]

		# New value = sigmoid( weightedSum( previous_neurons ) + bias )
		weightedSum = 0.0
		for i in range(0,len(inValues)):
			weightedSum = inValues[i] * self.weights[i] # Sumproduct of weights and values
		weightedSum = weightedSum + self.bias
		self.value = sigmoid(weightedSum) 


class neuralNetwork:
	def __init__(self, inLayer=28*28, layer1=16, layer2=16, outLayer=10):
		# Currently, these neural networks always have an input layer, 2 hidden layers, and an output layer
		self.layer1Neurons = [neuron(inLayer) for i in range(0,layer1)]
		self.layer2Neurons = [neuron(layer1) for i in range(0,layer2)]
		self.outputNeurons = [neuron(layer2) for i in range(0,outLayer)]

	# Get the new values of every neuron in the network, given the new input layer
	def feedForward(self, inputs):
		for i in self.layer1Neurons:
			i.update(inputs)
		for i in self.layer2Neurons:
			i.update(self.layer1Neurons)
		for i in self.outputNeurons:
			i.update(self.layer2Neurons)

	def getOutput(self):
		return([i.value for i in self.outputNeurons])

	def cost(self, correctOutput):
		cost = [(self.outputNeurons[i].value - correctOutput[i])**2 for i in range(0, len(self.outputNeurons))]
		cost = sum(cost)
		return(cost)

	def getwb(self):
		# Return formatted as [layer1Weights, layer1Bias, layer2Weights, layer2Bias, ... ]
		outArray = []
		for layer in [self.layer1Neurons, self.layer2Neurons, self.outputNeurons]:
			for neuron in layer:
				outArray += neuron.weights
				outArray.append(neuron.bias)
		return(outArray)




# Load abridged data
abridgedData = pickle.load(open('abridgedData.pickle', 'rb'))
pixels = abridgedData[0]

# Create a neural network
myNN = neuralNetwork()

# Get output from first training image
myNN.feedForward(pixels)
print(myNN.getOutput())

# Get correct answer from first training image
answers = [0.0 for i in range(0,10)]
answers[pixels[0]-1] = 1.0 

# Get cost for this output
myCost = myNN.cost(answers)
print(myCost)

# Get current weights and biases of this network
wb = myNN.getwb()
print(len(wb)) # This thing is too big, so only print out the length to make sure it's the right length

# Get gradient of this network (note this is the positive gradient and not the negative gradient)
grad = numpy.gradient(wb)
print(grad[:10])
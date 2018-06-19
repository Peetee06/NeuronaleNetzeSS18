
"""
Group: ---> Philipp Friedrich, Peter Trost <---

Your tasks:
Fill in your name.
Please complete the methods at the marked code blocks.
Please comment on the most important places.

Try your implmentation with
    MultyLayerAnn.py xor-input.txt xor-target.txt 1000 1 0.9 0]
    MultyLayerAnn.py or-input.txt or-target.txt 100 1 0.9 0]
    MultyLayerAnn.py digits-input.txt digits-target.txt 100 0.05 0.9 0]
"""
import numpy as np
import sys
import matplotlib.pyplot as plt


class MultiLayerANN:
    """
    The class MultiLayer implements a multi layer ANN with a flexible number of hidden layers.
    For learning backpropagation is used.
    """

    # the activation of the Biasneuron
    BIASACTIVATION= -1

    def __init__(self, *layerdimensions):
        """
        initializes a new MultiLayerANN object
        :param layerdimensions: each parameter describes the amount of neurons in the corresponding layer.
        E.g. MultiLayerANN(3,10,4) creates a network with 3 layers, 3 input neurons, 10 hidden neurons and 4 output neurons.
        """
        if len(layerdimensions) <2 :
            raise Exception("At least an input and output layer must be given")
        self._layerdimensions = layerdimensions

        #the netinput value for each non input neuron, each list element represents one layer
        self._netinputs = []
        # Type: list of np.arrays. The activation value for each non input neuron, each list element represents one layer
        self._activations = []
        # Type: list of np.arrays. The back propagation delta value for each non input neuron, each list element represents one layer
        self._deltas = []
        # Type: list of np.arrays. List of all weight matrices. Weight matrices are randomly initialized between -1 and 1
        self._weights= []
        # Type: list of np.arrays. List of all delta weight matrices. They are added to the corresponding weight matrices after each training step
        self._weights_deltas =[]

        prev_layersize= self._layerdimensions[0]
        for layersize in self._layerdimensions[1:] :
            self._netinputs.append(np.zeros(layersize))
            self._activations.append(np.zeros(layersize))
            self._deltas.append(np.zeros(layersize))

            # we use +1 to consider the bias-neurons
            # weights are chosen randomly (uniform distribution) between -1 and 1
            self._weights.append(np.random.rand(prev_layersize+1,layersize)*2-1)
            self._weights_deltas.append(np.zeros([prev_layersize + 1, layersize]))
            prev_layersize = layersize
        
        print(self._act_fun(0))

    def _act_fun(self,net_input):
        """
        :param net_input: single value or an array of net_inputs
        :return: sigmoid activation function
        """
        sigmoid = lambda x: 1/(1+math.exp(-x))
        if isinstance(net_input, np.ndarray):
            result = np.zeros(length(net_input))
            for i in range(0, length(net_input)):
                result[i] = sigmoid(net_input[i])
        else:
            result = sigmoid(net_input)
        return result

    def _act_func_derivative(self,net_input):
        """
        :param net_input: single value or an array of net_inputs
        :return: derivative of the sigmoid activation function
        """
		f = self._act_fun(net_input)
		if isinstance(f, np.ndarray):
			for i in range(0, length(f)):
				result = f[i]*(1-f[i])
		else:
			result = f*(1-f)
        return result

    def _predict(self, input):
        """
        calculates the output of the network for one input vector
        :param input: input vector
        :return: activation of the output layer
        """
		self._netinputs[0] = np.matmul(input, self._weights[0])
		self._activations[0] = self._act_fun(self._netinputs[0])
		for i in range(1, length(self._layerdimensions)):
			self._netinputs[i] = np.matmul(self._activations[i-1], self.weights[i])
			self._activations[i] = self._act_fun(self._netinputs[i])
        return []

    def _train_pattern(self, input, target,learningrate, momentumrate, weightdecay):
        """
        trains one input vector
        :param input (np.array): one input vector
        :param target (np.array):
        :param learningrate:
        :param momentumrate:
        :param weightdecay:
        :return: mean squared error over all output neurons
        """
        self._predict(input)


        return 0

    def train(self, inputs,targets,epochs,learningrate, momentumrate, weightdecay):
        """
        trains a set of input vectors. The error for each epoch gets printed out.
        In addition, the amount of correctly classiefied input vectors in printed
        :param inputs: list of input vectors
        :param targets: list of target vectors
        :param epochs:  number of training iterations
        :param learningrate: learningrate for the weight update
        :param momentumrate:  momentumrate for the weight update
        :param weightdecay:  weightdecay for the weight update
        :return: list of errors. One error value for each epoch. One error is the mean error over all input_vectors
        """
        errors= []
        for epoch in range(epochs):
            error=0
            for input ,target in zip(inputs,targets):
                error += self._train_pattern(input,target,learningrate,momentumrate,weightdecay)
            error /= len(inputs)
            errors.append(error)
            print("epoch: {0:d}   error: {1:f}".format(epoch,float(error)))

        print("final error: {0:f}".format(float(errors[-1])))

        # evaluate the prediction
        correct_predictions=0
        for input, target in zip(inputs, targets):
            # for one output use thresholding with 0.5
            if isinstance(target, float):
                correct_predictions += 1 if np.abs(self._predict(input)-target)>0.5 else 0

            # for multiple outputs choose the outputs with the highest value as predicted class
            else:
                prediction = self._predict(input)
                predicted_class = np.where(prediction==max(prediction))
                correct_predictions += 1 if  target[predicted_class] ==1 else 0
        print("correctly classified: {0:d} / {1:d}".format(correct_predictions,len(inputs)))

        return errors


def read_double_array(filename):
    """
    reads an np.array from the provided file.
    :param filename: path to a file
    :return: np.array of the matrix given in the file
    """
    with open(filename) as file:
        content = file.readlines()
    return np.loadtxt(content)


def main():
    # if len(sys.argv) != 7:
    #     print("usage: Perceptron <netinput-file> <teachingoutput-file> <iterations> <learningrate> <momentum> <weightdecay>")
    #    sys.exit(1)
    #args= sys.argv
    #args=[" " ,"digits-input.txt", "digits-target.txt", "100", "0.05","0.9","0"]
    #args=[" " ,"xor-input.txt", "xor-target.txt", "1000", "1","0.9","0"]
    args=[" " ,"or-input.txt", "or-target.txt", "1000", "1","0.9","0"]

    input_vectors = read_double_array(args[1])
    targets = read_double_array(args[2])
    epochs = int(args[3])
    learningrate = float(args[4])
    momentum = float(args[5])
    weightdecay = float(args[6])
    multilayerANN = MultiLayerANN(input_vectors.shape[1], 15, 1 if isinstance(targets[0], float) else targets.shape[1])

    errors= multilayerANN.train(input_vectors,targets,epochs,learningrate,momentum,weightdecay)

    plt.plot(errors, 'r')
    plt.ylabel('error')
    plt.xlabel('iteration')
    plt.show(block=True)


if __name__ == "__main__":
    main()






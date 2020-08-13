# standard library
import random

# asic math tools (Third-party library)
import numpy as np

# basic tools for working with files and directories
import os

def sigmoid(z):
    x = np.exp( z ) 
    return x / ( 1.0 + x )

def d_sigmoid(z):
    x = sigmoid( z )
    return x*(1-x)

def sigmoid_perceptron(activation, weight, bias):
    return sigmoid( np.dot(weight, activation) + bias )

class Network(object):
    
    def __init__(self, layers = None, name = None):
        if layers is None:    
            self.layers = 0
            self.biases = []
            self.weights = []
        elif name is None:
            self.layers = layers
            self.biases = []
            self.weights = []
        else:
            self.layers = layers
            self.load_network_from_files(name)
            
    def generate_random_network(self, size):
        self.layers  = len(size)-1
        self.biases  = []
        self.weights = []
        for i in range(self.layers):
            self.weights.append(np.random.rand(size[i+1],size[i]))
            self.biases.append(np.random.rand(size[i+1])-0.5)

    def load_network_from_files(self, name):
        self.biases = []
        self.weights = []
        
        path = os.getcwd()
        os.chdir(path + '/' + name)
        
        for i in range(0,self.layers):
            i += 1
            self.biases.append(np.loadtxt("bias."+str(i)+".csv", delimiter=",", ndmin=1))
            self.weights.append(np.loadtxt("weights."+str(i)+".csv", delimiter=",", ndmin=2))
        
        os.chdir(path)
        
    def save_network_to_files(self, name):
        path = os.getcwd()
        
        if not os.path.isdir('./'+name):
            os.mkdir(path + '/' + name)
        
        os.chdir(path + '/' + name)
        
        for i in range(0,self.layers):
            i += 1
            np.savetxt("bias."+str(i)+".csv", self.biases[i-1], delimiter=",")
            np.savetxt("weights."+str(i)+".csv", self.weights[i-1], delimiter=",")
        
        os.chdir(path)
    
    def feedforward(self, activation):
        for it in range(self.layers):
            forward = np.dot(self.weights[it], activation) + self.biases[it]
            activation = sigmoid(forward)
            
        return activation
    
    def feed_next(self, activation, layer):
        z = np.dot(self.weights[layer], activation) + self.biases[layer]
        n = sigmoid(z)
        return n,z
        
    def backpropagation(self, activation, expected_result):
        activations = [activation]
        zs = []
        for i in range(self.layers):
            a, z = self.feed_next(activations[i], i)
            activations.append(a)
            zs.append(z)
            
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b[-1] = (activations[self.layers] - expected_result) * d_sigmoid(zs[-1])
        grad_w[-1] = np.outer(grad_b[-1],activations[-2])
        for i in range(1,self.layers):
            grad_b[-i-1] = np.dot(self.weights[-i].transpose(), grad_b[-i])*d_sigmoid(zs[-i-1])
            grad_w[-i-1] = np.outer(grad_b[-i-1],activations[-i-2])
            
        return grad_b, grad_w
    
    def update_weights_and_bias(self, mini_batch, eta):
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_w = [np.zeros_like(w) for w in self.weights]
        
        for x, y in mini_batch:
            delta_b, delta_w = self.backpropagation(x, y)
            grad_b = np.add(grad_b,delta_b)
            grad_w = np.add(grad_w,delta_w)

        for i in range(self.layers):
            self.biases[i] -= eta/len(mini_batch)*grad_b[i]
            self.weights[i] -= eta/len(mini_batch)*grad_w[i]
            
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_result]
        return sum(int(x == y) for (x, y) in test_results)
    
    def stochastic_update(self, full_batch, mini_batch_length, eta, epochs, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(full_batch)
        for j in range(epochs):
            random.shuffle(full_batch)
            mini_batches = [full_batch[k:k+mini_batch_length] for k in range(0, n, mini_batch_length)]
            for mini_batch in mini_batches:
                self.update_weights_and_bias(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))

# standard library
import random

# Basic math tools (Third-party library)
import numpy as np

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

    def load_network_from_files(self, name):
        self.biases = []
        self.weights = []
        
        for i in range(0,self.layers):
            i += 1
            self.biases.append(np.loadtxt(name+"."+"bias."+str(i)+".csv", delimiter=",", ndmin=1))
            self.weights.append(np.loadtxt(name+"."+"weights."+str(i)+".csv", delimiter=",", ndmin=2))
            
    def save_network_to_files(self, name):
        for i in range(0,self.layers):
            i += 1
            np.savetxt(name+"."+"bias."+str(i)+".csv", self.biases[i-1], delimiter=",")
            np.savetxt(name+"."+"weights."+str(i)+".csv", self.weights[i-1], delimiter=",")
                
    
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
            
        grad_b = np.array([np.zeros(b.shape) for b in self.biases])
        grad_w = np.array([np.zeros(w.shape) for w in self.weights])
        grad_b[-1] = (activations[self.layers] - expected_result) * d_sigmoid(zs[-1])
        grad_w[-1] = np.outer(grad_b[-1],activations[-2])
        for i in range(1,self.layers):
            grad_b[-i-1] = np.dot(self.weights[-i].T, grad_b[-i])*d_sigmoid(zs[-i-1])
            grad_w[-i-1] = np.outer(grad_b[-i-1],activations[-i-2])
            
        return grad_b, grad_w
    
    def update_weights_and_bias(self, mini_batch, eta):
        grad_b = np.array([np.zeros(b.shape) for b in self.biases])
        grad_w = np.array([np.zeros(w.shape) for w in self.weights])
        
        for approx, exact in mini_batch:
            delta_b, delta_w = self.backpropagation(approx, exact)
            grad_b = np.add(grad_b,delta_b)
            grad_w = np.add(grad_w,delta_w)

        for i in range(self.layers):
            self.biases[i] -= eta/len(mini_batch)*grad_b[i]
            self.weights[i] -= eta/len(mini_batch)*grad_w[i]
    
    def stochastic_update(self, full_batch, mini_batch_length, eta):
        mini_batch_sample = random.sample(range(len(full_batch)),mini_batch_length)
        mini_batch        = [full_batch[i] for i in mini_batch_sample]
        self.update_weights_and_bias(mini_batch, eta)

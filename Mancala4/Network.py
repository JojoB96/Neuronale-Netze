# standard library
import random

# asic math tools (Third-party library)
import numpy as np

# basic tools for working with files and directories
import os

def sigmoid(z):
    x = np.exp( -z ) 
    return 1 / ( 1.0 + x )

def d_sigmoid(z):
    x = sigmoid( z )
    return x*(1-x)

def ReLU(z):
    if z is int:
        x = max(0,z)
    else:
        x = [max(0,a[0]) for a in z]
        
    return np.reshape(x,(len(x),1))

def d_ReLU(z):
    if z is int:
        x = 1 * (z > 0)
    else:
        x = [1. * (a[0] > 0) for a in z]
    return np.reshape(x,(len(x),1))

def MeanValue(a,y):
    temp =  a - y
    return (1. / len(y)) * temp

def HyperbolicTangent(z):
    x = np.sinh(z)/np.cosh(z)
    return x

def d_HyperbolicTangent(z):
    x = 2./(np.cosh(2.*z) + 1.)
    return x

def CrossEntropy(a,y):
    a = np.clip(a,1e-10,1-1e-10)
    return (1. / len(y)) * (np.divide(-y,a)+np.divide(np.ones_like(y)-y,np.ones_like(a)-a))

def sigmoid_perceptron(activation, weight, bias):
    return sigmoid( np.dot(weight, activation) + bias )

class Network(object):
    
    def __init__(self, layers = None, name = None, act_func = "sigmoid", cost = "MeanValue"):
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
            
        if act_func is "sigmoid":
            self.act_func   = sigmoid
            self.d_act_func = d_sigmoid
        elif act_func is "ReLU":
            self.act_func   = ReLU
            self.d_act_func = d_ReLU
        elif act_func is "HyperbolicTangent":
            self.act_func   = HyperbolicTangent
            self.d_act_func = d_HyperbolicTangent
        if cost is "CrossEntropy":
            self.cost      = CrossEntropy
        elif cost is "MeanValue":
            self.cost      = MeanValue
            
    def generate_random_network(self, size):
        self.layers  = len(size)
        self.biases  = [np.random.randn(y,1) for y in size[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(size[:-1], size[1:])]
        

    def load_network_from_files(self, name):
        self.biases = []
        self.weights = []
        
        path = os.getcwd()
        os.chdir(path + '/' + name)
        
        for i in range(1,self.layers):
            self.biases.append(np.loadtxt("bias."+str(i)+".csv", delimiter=",", ndmin=2))
            self.weights.append(np.loadtxt("weights."+str(i)+".csv", delimiter=",", ndmin=2))
        
        os.chdir(path)
        
    def save_network_to_files(self, name):
        path = os.getcwd()
        
        if not os.path.isdir('./'+name):
            os.mkdir(path + '/' + name)
        
        os.chdir(path + '/' + name)
        
        for i in range(1,self.layers):
            np.savetxt("bias."+str(i)+".csv", self.biases[i-1], delimiter=",")
            np.savetxt("weights."+str(i)+".csv", self.weights[i-1], delimiter=",")
        
        os.chdir(path)
    
    def feedforward(self, activation):
        for b, w in zip(self.biases, self.weights):
            activation = self.act_func(np.dot(w, activation)+b)
        return activation
    
    def feed_next(self, activation, layer):
        z = np.dot(self.weights[layer], activation) + self.biases[layer]
        n = self.act_func(z)
        return n,z
        
    def backpropagation(self, activation, expected_result):
        activations = [activation]
        zs = []
        for i in range(self.layers - 1):
            a, z = self.feed_next(activations[i], i)
            activations.append(a)
            zs.append(z)
        #print(activations)
        #input()
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_w = [np.zeros_like(w) for w in self.weights]
        
        delta      = self.cost(activations[-1],expected_result) * self.d_act_func(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2,self.layers):
            z = zs[-l]
            sp = self.d_act_func(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta,activations[-l-1].transpose())
            
        return (grad_b, grad_w)
    
    def update_weights_and_bias(self, mini_batch, eta):
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_w = [np.zeros_like(w) for w in self.weights]
        
        for x, y in mini_batch:
            delta_b, delta_w = self.backpropagation(x, y)
            grad_b = [gb + db for gb, db in zip(grad_b, delta_b)]
            grad_w = [gw + dw for gw, dw in zip(grad_w, delta_w)]

        self.biases  = [b - (eta/len(mini_batch))*gb for b, gb in zip(self.biases, grad_b)]
        self.weights = [w - (eta/len(mini_batch))*gw for w, gw in zip(self.weights, grad_w)]
            
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
            #if test_data:
            #    print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data),n_test))
            #else:
            #    print("Epoch {0} complete".format(j))

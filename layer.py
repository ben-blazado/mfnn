#--- layer.py
import numpy as np

class Layer():
    
    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.outputs = None
        self.errors = None
        self.name = "Layer"
        return
        
    def compile(self, prev_layer):
        self.input_size = prev_layer.output_size
        self.output_size = self.input_size
        return
    
    def update_outputs(self, inputs):
        self.outputs = inputs
        return
    
    def update_errors(self, errors):
        self.errors = errors
        return
    
    def update_weights(self, learning_rate, batch_size):
        pass
    
    def init_weight_steps(self):
        pass
    
    def summary(self):
        print (self.name + ":", (self.input_size, self.output_size))
        

class Dense(Layer):
    
    def __init__(self, output_size, input_size=None, weights=None):

        Layer.__init__(self)
        self.mean = 0.0
        self.std_dev = 0.1
        self.weights = weights
        self.biases = None
        self.output_size = output_size
        self.input_size = input_size
        self.name = "Dense"
        self.batch_size = 1

        return
    
        
    def _init_weights(self, input_size):
        
        if self.weights is None:
            shape_weights = (input_size, self.output_size)
            self.weights = np.random.normal(self.mean, self.std_dev, shape_weights)
            self.biases  = np.random.normal(self.mean, self.std_dev, self.output_size)
            
        return
    
    def compile(self, prev_layer):

        if prev_layer is not None:
            self.input_size = prev_layer.output_size
        self._init_weights(self.input_size)
        return

        
    def update_outputs(self, x):
        
        # x.shape should be (1, input_size)
        # shape of weights is (input_size, output_size)
        # self.outputs = np.dot(x, self.weights) + self.biases
        # shape of outputs is (1, output_size)        
        # we save x (the layer inputs) to self.inputs...
        # ... because it will be used to update delta_w in update_errors()
        self.inputs = x    
        self.outputs = np.dot(self.inputs, self.weights) 
        return 

        
    def update_errors(self, errors):
        # self.error back propagated to update prev layer
        self.delta_w += errors * self.inputs.T
        self.errors = np.dot (errors, self.weights.T)

        return
    
    def init_weight_steps(self):
        self.delta_w = np.zeros(self.weights.shape)
        return
    
    def update_weights(self, learning_rate, sample_size):
        self.weights += learning_rate * self.delta_w / sample_size
        return
    
        
class Activation(Layer):
    '''default activation layer is a linear activation'''
    
    def __init__(self):
        Layer.__init__(self)
        self.name = "Activation(Default Linear)"
        return
    
    def compile(self, prev_layer):
        self.input_size = prev_layer.output_size
        self.output_size = self.input_size
        
        return
    
    def f(self, x):
        #override f in descendants
        #linear activation is f(x) = x
        return x
    
    def dfdx(self):
        # override dfdx (derivative of f(x) wrt x) in descendents
        # derivative of f(x) is f'(x) = 1 
        # dfdx() is used in the call to update_errors
        return 1
    
    def update_outputs(self, x):
        # linear activation 
        self.outputs = self.f(x)
        return
    
    def update_errors(self, errors):
        self.errors = errors * self.dfdx()
        return
    
    def update_weights(self, learning_rate, batch_size):
        # activation layers have no weights to update
        pass
        
    
    
class Sigmoid(Activation):
    
    def __init__(self):
        Activation.__init__(self)
        self.name = "Sigmoid Activation"
        return
    
    def f(self, x):
        # f(x) = 1 / (1 + exp(x))
        return 1.0 / (1.0 + np.exp(-x))
    
    def dfdx(self):
        # for sigmoid function, f'(x) = f(x) * (1 - f(x))
        return self.outputs * (1 - self.outputs)
    

#--- TODO: fix leaky rectified linear unit activation
class LRelu(Activation):
    #--- DON't USE YET
    
    def __init__(self):
        Activation.__init__(self)
        self.name = "Leaky ReLU"
        self.threshold = 0.1
        return
    
    def f(self, x):
        #--- x.shape (1, num_outputs)
        f = np.array([[max (m, self.threshold) for m in x[0]]])
        assert (len(f.shape) == 2)
        return f
    
    def dfdx(self):
        #--- self.outputs is a vector
        #--- self.outputs.shape is (1, num_ouputs)
        dfdx = np.array([[1.0 if o > self.threshold else 0.01 for o in self.outputs[0]]])
        assert (len(dfdx.shape) == 2)
        return dfdx
    
            

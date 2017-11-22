#--- neural_network.py
import numpy as np
from time import sleep   # TODO: Delete after debug

class NeuralNetwork():
    
    def __init__(self):
        
        self.layers = []
        self.lowest_loss = float('Inf')
        return
    
    def add(self, layer):
        self.layers.append (layer)
        return
    
    def compile(self):
        prev_layer = None
        for layer in self.layers:
            layer.compile(prev_layer)
            prev_layer = layer
        return
    
    def summary(self):
        print ("---------------------------------------")        
        print ("Neural network layers (inputs, outputs)")
        print ("---------------------------------------")        
        for layer in self.layers:
            layer.summary()
        print ("---------------------------------------")
        return
        
    def update_outputs(self, inputs):

        #--- one forward pass
        #--- update first hidden layer [0]
        self.layers[0].update_outputs(inputs)
        
        #--- update all remaining layers [1:]
        prev_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.update_outputs(prev_layer.outputs)
            prev_layer = layer
        return 
    
    def update_errors(self, labels):

        #--- back propagate output errors
        #--- output layer is last layer [-1]
        output_error = np.array(labels - self.layers[-1].outputs, ndmin=2)
        self.layers[-1].update_errors(output_error)
        
        #--- backpropagate error by iterating the reversed order of layers
        prev_layer = self.layers[-1]
        for layer in (self.layers[::-1])[1:]:
            layer.update_errors(prev_layer.errors)
            prev_layer = layer
        return

    def update_weights(self, learning_rate, batch_size):
        for layer in self.layers:
            layer.update_weights(learning_rate, batch_size)
        return
    
    
    def _init_training_batches(self, inputs, targets, batch_size):
        num_samples = len(inputs)
        self.training_batches=[]
        
        for start in range(0, num_samples, batch_size):
            end            = min (start + batch_size, num_samples)
            features       = inputs[start:end]
            labels         = targets[start:end]
            training_batch = (features, targets)
            self.training_batches.append(training_batch)
        return 
    
    def create_random_batch(self, inputs, targets, batch_size=32):
        
        indices = np.random.choice(len(inputs), size=batch_size)
        
        return inputs[indices], targets[indices]
    
    
    def _init_weight_steps(self):
        for layer in self.layers:
            layer.init_weight_steps()
    
    
    def predict_on_batch(self, batch_features):
        batch_predictions = np.array([])
        for features in batch_features:
            self.update_outputs(features)
            batch_predictions = np.append(batch_predictions, self.layers[-1].outputs)
        return batch_predictions
    
    
    def test_on_batch(self, batch_features, batch_labels):
        batch_predictions = self.predict_on_batch(batch_features)
        sq_errors = [(labels - predictions) ** 2 for labels, predictions in zip(batch_labels, batch_predictions)]
        loss = np.mean(sq_errors)
        return loss
    
    
    def train_on_batch(self, batch_features, batch_labels, learning_rate):
        
        #--- len (batch_features) == len (batch_labels)
        self._init_weight_steps()
        
        #--- process each row of features and labels in batch
        for features, labels in zip(batch_features, batch_labels):
            self.update_outputs(features)
            self.update_errors(labels)
            
        #--- all rows processed; update weights and evaluate model
        self.update_weights(learning_rate, batch_size=len(batch_features))
        batch_loss = self.test_on_batch(batch_features, batch_labels)
        
        return batch_loss
        
    
    #--- 
    def fit (self, inputs, targets, epochs, learning_rate, batch_size, verbose=1, validation_data=None):
        '''
        inputs:  numpy array of shape (num_inputs, num_features)
        targets: numpy array of shape (num_inputs, num_labels) i.e. target.shape[0] == inputs.shape[0]
        '''
        
        losses = []
        validate = validation_data is not None
        
        if validate:
            #--- unpack validation data
            validation_inputs, validation_targets = validation_data
            validation_losses = []
            
        for e in range(epochs):
            #--- process epoch e
            
            #--- 15nov17: we needed to select a random sample to prevent overfitting
            batch_features, batch_labels = self.create_random_batch(inputs, targets, batch_size)
            batch_loss     = self.train_on_batch(batch_features, batch_labels, learning_rate)
            
            losses.append(batch_loss)
            
            if validate:
                validation_loss = self.test_on_batch(validation_inputs, validation_targets)
                validation_losses.append(validation_loss)
            
            if (verbose != 0):
                status = {"epoch" : e + 1, 
                          "epochs": epochs, 
                          "loss"  : batch_loss,
                          "vloss" : validation_loss if validate else "N/A"}
                print ("Epoch {epoch:d} of {epochs:d} | Loss: {loss:f} | Validation Loss: {vloss:f}".format(**status), end="\r")
                sleep(0.0005)
                
        return losses, validation_losses
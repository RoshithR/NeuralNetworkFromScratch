from nnfs.datasets import spiral_data
import numpy as np
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights)+self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(inputs,0)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs<=0]=0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y_true):
        sample_loss = self.forward(output, y_true)
        data_loss = np.mean(sample_loss)
        return data_loss

class Loss_CrossEntropy(Loss):
    def forward(self, y_pred,y_true):
        #clipping y_pred to avoid logarithm of 0 error
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[[range(len(y_pred))],y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        neg_loss = -np.log(correct_confidences)
        return neg_loss
    def backward(self, dvalues, y_true):
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true/dvalues
        #Normalize gradients
        self.dinputs = self.dinputs/len(dvalues)

# Softmax classifier combined Softmax activation & cross entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CrossEntropy()
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #If labels are one-hot encoded turn them into discrete values
        if len(y_true.shape)==2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples),y_true] -= 1
        #Normalize gradients
        self.dinputs = self.dinputs/samples

class Optimizer_SGD:
    #Initiaize optimizer
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rae = learning_rate
        self.decay = decay
        self.iterations = 0
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rae = self.learning_rate * (1./(1.+ self.decay *self.iterations))
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.current_learning_rae * layer.dweights
        layer.biases += self.current_learning_rae * layer.dbiases
    # Call once after parameter update
    def post_update_params(self):
        self.iterations += 1

if __name__=="__main__":
    # Create dataset
    X,y = spiral_data(samples=1000,classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Dense_Layer(2,3)

    # Create ReLU activation
    activation1 = Activation_ReLU()

    # Create second Dense layer with 3 input features (output of previous layer) and 3 output values
    dense2 = Dense_Layer(3,3)

    # Create Softmax activation
    # activation2 = Activation_Softmax()
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


    # loss_fn = Loss_CrossEntropy()

    #Create optimizer
    optimizer = Optimizer_SGD(decay=1e-2)

    for epoch in range(10001):
        #forward

        # Forward pass through first dense layer
        dense1.forward(X)

        # Forward pass through activation function
        activation1.forward(dense1.output)

        # Forward pass through second dense layer
        dense2.forward(activation1.output)

        # Forward pass through Softmax activation function
        # activation2.forward(dense2.output)
        # loss= loss_fn.calculate(activation2.output,y)
        loss = loss_activation.forward(dense2.output,y)
        print('loss', loss)
        # predictions = np.argmax(activation2.output, axis=1)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape)==2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        if not epoch % 100:
            print(f'epoch: {epoch},' +
                  f'acc: {accuracy:.3f},'+
                  f'loss: {loss:.3f},'+
                  f'lr: {optimizer.current_learning_rae}')

        # Backward pass
        loss_activation.backward(loss_activation.output,y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()






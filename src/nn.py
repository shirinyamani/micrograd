import random
from engine import Value

class Neuron:
    def __init__(self, nin): #number of neurons
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # n = wx + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), start=self.b)
        out = act.tanh()
        return out

    def parameters(self): #we wanna a convenient function to nudge the w, b based on the gradient info
        return self.w + [self.b]
        

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] #a layer is a list of neurons and how many neurons (nout)

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        # ps = neuron.parameters()
        # params.extend(ps)
        #return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts #intire neurons in a single layer
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] #i + i+i current layer and the next one 

    def __call__(self, x): #call layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x


    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        
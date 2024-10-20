from abc import ABC, abstractmethod
import numpy as np
class BaseLayer(ABC):
    def __init__(self, *weights):
        self.weights = weights

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'backward') and 0:
            cls.backward = io(level='trace')(cls.backward)
            cls.forward = io(level='trace')(cls.forward)

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)

    @abstractmethod
    def forward(self, x):
        pass

class Sequential(BaseLayer):
    def __init__(self, *layers):
        self.layers = layers
        super().__init__()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, loss):
        for layer in self.layers[::-1]:
            loss = layer.backward(loss)

    def update(self, *a, **kw):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(*a, **kw)

class Linear(BaseLayer):
    def __init__(self, ni, no):
        self.w = np.random.randn(ni, no)
        self.b = np.random.randn(no,)
        super().__init__(self.w, self.b)

    def forward(self, x):
        self.cache = x
        o = x @ self.w + self.b
        return o

    def backward(self, loss):
        x = self.cache
        self.dw = x.T @ loss
        self.db = loss
        return loss @ self.w.T

    def update(self, alpha):
        # import ipdb; ipdb.set_trace()b
        self.w -= alpha*self.dw
        self.b -= alpha*self.db.mean()

    def __repr__(self):
        return f'linear layer (y = \n{self.w}x + {self.b})'

class ReLU(BaseLayer):
    def forward(self, x):
        self.cache = x
        return x * (x > 0)

    def backward(self, loss):
        x = self.cache
        return loss * (x > 0)

class Sigmoid(BaseLayer):
    def forward(self, x):
        o = 1 / (1 + np.exp(-x))
        self.cache = o
        return o

    def backward(self, loss):
        o = self.cache
        return loss * o * (1 - o)

class BCE(BaseLayer):
    eps = 1e-7
    def forward(self, y, y_hat):
        self.cache = (y_hat, y)
        loss = - (y*np.log(y_hat+self.eps) + (1-y)*np.log(1-y_hat+self.eps))
        return loss.mean()

    def backward(self, loss):
        y_hat, y = self.cache
        return - (y/y_hat - (1-y)/(1-y_hat)) * loss

class MSE(BaseLayer):
    def forward(self, y, y_hat):
        self.cache = (y_hat, y)
        return ((y - y_hat)**2).mean()

    def backward(self, loss):
        y_hat, y = self.cache
        return (y_hat - y) * loss

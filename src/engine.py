import math
class Value:
    def __init__(self, data, _children=(), _op='', label=''): #children to kinda relate them to one another!
        self.data = data
        self._children = set(_children) #children for understanding the created value is from what, e.g. d is from addition of a,b
        self._op = _op #from what operation 
        self.label = label #to create a label for the dat values
        self.grad = 0.0 #initial set to zero, meaning slight change in the variables does not impact the loss function
        self._backward = lambda: None #chain rule at each node empty by default bc of the some of the biases #for leaf node
        
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other) #a+1
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad #local der * global der (der of final w respect to that data)
            other.grad += 1.0 * out.grad #copied
        out._backward = _backward
        return out



    def __pow__(self, other):
        assert isinstance(other, (int,float)), "inly supporting int/float powers"
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            self.grad +=  other * (self.data **(other-1)) * out.grad 
            
        out._backward = _backward

        return out
        
    def __mul__(self, other, _op='*'):
        other = other if isinstance(other, Value) else Value(other) #aa*2
        out = Value(self.data * other.data, (self,other), '*')
        def _backward():
            self.grad +=  other.data * out.grad #local der * global der (der of final w respect to that data)
            other.grad += self.data * out.grad # we wanna chain the out.grad into other.grad 
            
        out._backward = _backward

        return out

    def __rmul__(self, other): #2 * a == a * 2 (a.__mul__(2)):
        return self * other 


    def __truediv__(self, other): #self / other
        return self * other**-1

    def __neg__(self): #-self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,),'exp')

        def _backward():
            self.grad +=  out.data * out.grad #deriv e*x is e*x that we already calculated inside the out 
        out._backward = _backward

        return out

    
    def tanh(self): #dont need atomic function of, up to us how complicated 
        x = self.data
        t = (math.exp(2*x)-1)/ (math.exp(2*x)+1)
        out = Value(t, (self, ),'tanh') #tuple is the children of this node 

        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v) #mark as visied 
                for child in v._children:
                    build_topo(child)
                topo.append(v) #after all children are processed go and add yourself 
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

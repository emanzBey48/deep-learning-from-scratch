import matplotlib.pyplot as plt
class Value:
    def __init__(self, data, _children=(), _op='',label=''):
        self.data= data
        self.grad=0.0
        self._backward = lambda:None
        self._prev= set(_children)
        self._op= _op
        self.label=label

    def __truediv__(self, other):
        return self * (other**-1)

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance (other, Value) else Value(other)
        out= Value(self.data + other.data, (self,other),'+' )
        def _backward():
            self.grad+=1.0 * out.grad
            other.grad+=1.0* out.grad
        out._backward=_backward
        
        return out
    def __mul__(self, other):
            other = other if isinstance (other, Value) else Value(other)
            out= Value(self.data * other.data,(self,other),'*')
            def _backward():
                 self.grad+=other.data*out.grad
                 other.grad+=self.data*out.grad
            out._backward=_backward
            return out

    def __pow__(self,other):

        assert isinstance (other , (int,float))
        out=Value(self.data**other,(self,), f'**{other}')
        def _backward():
            self.grad+=other * (self.data **(other -1)) * out.grad
        out._backward=_backward
        return out
    def __sub__(self, other):
        return self +(-other)
    def __neg__(self):
        return self * -1
    
    def tanh(self):
            import math
            x=self.data
            t=(math.exp(2*x)-1)/ (math.exp(2*x)+1)
            out=Value(t ,(self, ),'tanh')
            def _backward():
              self.grad+=(1- t**2) * out.grad
            out._backward=_backward
            return out
    def __radd__(self, other): # other + self
        return self + other
    def exp(self):
        
        import math
        x=self.data
        out=Value(math.exp(x), (self, ),'exp')

        def _backward():
             self.grad+= out.data*out.grad
        out._backward=_backward
        return out

    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)       
        build_topo(self)
        self.grad=1.0
        for node in reversed (topo):
            node._backward()    

    def __rmul__(self, other):
         return self*other     


a=Value(2.0, label='a')
b=Value(3.0, label='b')
c=Value(100.0 , label='c')
e=a*c; e.label='e'
f=Value(-2.0 , label='f')
d=e + b ; d.label='d'
L=d*f; L.label='L'
# print(L)
#print(d)
#print(d._prev)
#print(d._op)
#print(a+b)
#print(b*c)
d.grad=-2.0
f.grad=205.0
e.grad=-2.0
b.grad=-2.0
a.grad=-200.0
c.grad=-4.0
L.grad=1.0


#inputs
x1=Value(2.0,label='x1')
x2=Value(0.0,label='x2')
#weights
w1=Value(-3.0,label='w1')
w2=Value(1.0,label='w2')
#bias
b=Value(6.8813,label='b')
x1w1=x1*w1; x1w1.label='x1w1'
x2w2=x2*w2; x2w2.label='x2w2'
x1w1x2w2= (x1w1 + x2w2);x1w1x2w2.label='x1w1 + x2w2'
n=x1w1x2w2+b;n.label='n'
o=n.tanh()

o.grad=1.0
o.backward()
print(n.grad)
'''
do/dn= 1-tanh(n)**2
do/dn= 1-o**2
tanhh=(1-o.data**2)
n.grad=tanhh
x1w1x2w2.grad=tanhh
b.grad=tanhh
x1w1.grad=tanhh
x2w2.grad=tanhh
x1.grad=w1.data*tanhh
w1.grad=x1.data*tanhh
x2.grad=w2.data*tanhh
w2.grad=x2.data*tanhh 
'''
'''
o._backward()
n._backward()
x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()
'''
'''
o.backward()
print(x1.grad)

'''
'''
a=Value(2.0)
b=Value(2.0)
print(a-b)
print(a/b)'''


#inputs
x1=Value(2.0,label='x1')
x2=Value(0.0,label='x2')
#weights
w1=Value(-3.0,label='w1')
w2=Value(1.0,label='w2')
#bias
b=Value(6.8813,label='b')
x1w1=x1*w1; x1w1.label='x1w1'
x2w2=x2*w2; x2w2.label='x2w2'
x1w1x2w2= (x1w1 + x2w2);x1w1x2w2.label='x1w1 + x2w2'
n=x1w1x2w2+b;n.label='n'

nn=2*n
e=nn.exp()
o=(e-1)/(e+1)
o.backward()
print(n.grad)
'''
import torch
x1= torch.Tensor([2.0]).double()        ; x1.requires_grad = True
x2= torch.Tensor([0.0]).double()        ; x2.requires_grad = True
w1= torch.Tensor([-3.0]).double()       ; w1.requires_grad = True
w2= torch.Tensor([1.0]).double()        ; w2.requires_grad = True
b= torch.Tensor([6.88137358]).double()  ; b.requires_grad = True

n=((x1*w1)+(x2*w2))+b;n.label='n'
o=torch.tanh(n)
print(o.data.item())
o.backward()
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())'''

class Neuron:
    def __init__(self,nin):
        import random
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act= sum((wi*xi for wi ,xi in zip(self.w,x)), self.b)
        out=act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self,nin,nout):
        self.neurons= [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs= [n(x) for n in self.neurons]
        return outs [0] if len(outs)==1 else outs
    def parameters(self):
       return [p for neuron in self.neurons for p in neuron.parameters()]
        
class MLP:
    def __init__(self,nin,nouts):
        sz=[nin] + nouts
        self.layers= [Layer(sz[i],sz[i+1]) for i in range (len(nouts))]
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return[p for layer in self.layers for p in layer.parameters()]

x=[2.0 , 3.0 , -1.0]
n = MLP(3,[4,4,1])
print(n(x))
        

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

#print(ypred)



for k in range(20):
    ypred= [n(x) for x in xs]
    loss=sum([(yout-ygt)**2 for ygt,yout in zip(ys,ypred)])
    for p in n.parameters():
        p.grad=0.0
    loss.backward()

    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)

print(ypred)
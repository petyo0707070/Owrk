import numpy as np
import matplotlib.pyplot as plt


#### Parameters
## Number of observations
K = 200
## Number of test observations
Ktest = 100
## Input dimenson
m = 1
## Output dimension
d = 1

## Number of digits shown
dig = 4

## GD parameters
## Number of gradient descent steps
N = 50
## Initial learning rate 
eta1 = 1
## Update learning rate according to eta = eta1 / step ??
dynamic = True



#### K Random observations of input respectively output dimension
x = np.random.normal(0,2,(K,m))
y = x[:,0:d] + np.random.normal(1,1,(K,d))
if m == 1:
    if d == 1:
        plt.plot(x,y,'o')
        plt.title("Obtained Data: x against y")
        plt.xlabel('x')  # Label for x-axis
        plt.ylabel('y')  # Label for y-axis        
        plt.show()
        
        
        
#### Model setup        
## Loss function
# L expects matrices y_pred, y where it computes 
# a vector with the squared Euclidean distances of the rows
# Each row is treated separately
def L(y_pred,y):
    return np.sum( (y_pred-y)**2, axis=1) ## scalar product with itself


## Initial setup: Here for xA +b = y
A = np.random.normal(0,1,(m,d))
b = np.random.normal(0,1,(d,))


#### Netowrk, simple affine network
def g(x,A,b):
    return np.dot(x,A) + b


## We like to minimise the total loss f: 
def f(x,y,A,b):
    y_pred = g(x,A,b)
    return np.mean(L( y_pred , y ))

## Numerical gradient of f, eps is the precision in the derivative approximation
def df(x,y,A,b,eps=0.0001):
    dA = np.zeros(np.shape(A)) ## initialise
    db = np.zeros(np.shape(b)) ## initialise
    ## The following is a highly non-optimal way of computing the gradient:
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1
        db[i] = (f(x,y,A,b+eps*e) - f(x,y,A,b-eps*e)) / (2*eps)
        for j in range(m):
            e = np.zeros((m,d))
            e[j,i] = 1
            dA[j,i] = (f(x,y,A+eps*e,b) - f(x,y,A-eps*e,b)) / (2*eps)
    return(dA,db)
    


print("\nStart at:\n",np.round(A,dig),"\nand intercept:",np.round(b,dig),"\nwith initial objective value is:",np.round(f(x,y,A,b),dig))


## Now, we do the actual gradient descent:
for i in range(N):
    eta = eta1/(i+1)  ## always dynamic 
    (dA,db) = df(x,y,A,b)
    #print("\n",A,"\n",eta*dA) ## for debugging purpose
    A = A - eta*dA  ## Gradient descent update for A
    b = b - eta*db  ## Gradient descent update for b




#### Model evaluation
print("\nEnded at:\n",np.round(A,dig),"\nand intercept:",np.round(b,dig),"\nwith final objective value (on-sample) is:",np.round(f(x,y,A,b),dig))



plt.plot([np.min(y),np.max(y)],[np.min(y),np.max(y)])
for i in range(d):
    plt.plot(g(x,A,b)[:,i],y[:,i],'o')
plt.title("Dots: True value against prediction;  Line: True value against true value")
plt.xlabel('y_pred = x A + b')  # Label for x-axis
plt.ylabel('y')  # Label for y-axis        
plt.show()



## K Random observations of input respectively output dimension
xtest = np.random.normal(0,2,(Ktest,m))
ytest = xtest[:,0:d] + np.random.normal(1,1,(Ktest,d))

## Test plot:
plt.plot([np.min(ytest),np.max(ytest)],[np.min(ytest),np.max(ytest)])
for i in range(d):
    plt.plot(g(xtest,A,b)[:,i],ytest[:,i],'o')
plt.title("Dots: True value against prediction;  Line: True value against true value")
plt.xlabel('y_pred')  # Label for x-axis
plt.ylabel('y')  # Label for y-axis        
plt.show()

print("\nObjective value (out-of-sample) is:",np.round(f(xtest,ytest,A,b),dig))


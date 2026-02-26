---
layout: page
title: Building blocks for a simple model
---


# Construction

To build a model we will need the following libraries
```python
import torch
from torch import nn
```

The logic is the following: we construct a model, i.e. a child `class` of the parent `nn.Module` with a specific arcitecture, and then create an instance of this `class`. Let's illustrate it with a very simple architecture conveient for classifying the `MNIST Dataset`. 

The architecture is depicted in the figure below

{% mermaid %}
flowchart LR;
    A["Input<br>(64, 1, 28, 28)"] e1@==>B["Hidden 1 <br>(64, 512)"]e2@==>C["Hidden 2 <br>(64, 512)"]e3@==>D["Output <br>(64, 10)"];
    e1@{ animate: true}
    e2@{ animate: true}
    e3@{ animate: true}
{% endmermaid %}






<!-- ```text
Input (batch_size, 1, 28, 28) -> 
Fully connected (fc1) layer (with n_neurons1 neurons) -> ReLU() -> 
Fully connected (fc2) layer (with n_neurons2 neurons) -> ReLU() -> 
Output (batch_size, 10)
``` -->

There are many ways to implement this architecture. We present two options. Explicit: 

```python
class MNIST_FC_2(nn.Module): # MNIST_FC_2L class inherits from the PyTorch class nn.Module
    def __init__(self,n_neurons1,n_neurons2): # This is the standard way to initialize a class
        super.__init__() # Needed to initialize everything from the inherited parent class, i.e. nn.Module
        self.flatten = nn.Flatten() # nn.Flatten() is defined by default with start_dim = 1, meaning it does not collapse the 0th dimension (batch index)
        self.fc1 = nn.Linear(1*28*28,n_neurons1) # Linear map for each element in the batch from 1x28x28 to n_neurons1
        self.fc2 = nn.Linear(n_neurons1,n_neurons2)
        self.fc3 = nn.Linear(n_neurons2,10)

    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)   # ReLU function from torch.nn.functional
        x = self.fc2(x)
        x = nn.functional.relu(x)
        logits = self.fc3(x)
        return logits
```

If we create an instance and print it we will receive
```python
model = MNIST_FC_2(512,512)
print(model)
```

<span style="color: grey;">Out:</span>
```text
MNIST_FC_2L(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=784, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=10, bias=True)
)
```

Packaged:

```python
class MNIST_FC_2(nn.Module):
    def __init__(self,n_neurons1,n_neurons2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.box = nn.Sequential(
            nn.Linear(1*28*28,n_neurons1),
            nn.ReLU(),
            nn.Linear(n_neurons1,n_neurons2),
            nn.ReLU(),
            nn.Linear(n_neurons2,10)
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.box(x)
        return logits
```

In this case creating an instance and printing it leads to
```python
model = MNIST_FC_2(10,20)
print(model)
```

<span style="color: grey;">Out:</span>
```text
FC_MNIST_2(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (box): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

The difference is that `ReLU()` in the latter case is considered a layer rather than just an operation.

Now we can take a batch from a `train_dataloader` (see [Datasets and DataLoaders](/ML/dataloaders/) for how to construct Dataloaders) and pass it to the `model()` method

```python
X, y = next(iter(train_dataloader))
print(X.shape)
print(y.shape)
print(model(X).shape)
```
<span style="color: grey;">Out:</span>
```text
torch.Size([64, 1, 28, 28])
torch.Size([64])
torch.Size([64, 10])
```

Note that we do not call explicitly `model.forward()`.



#### Parameters of the model

The parameters of a model can be accessed in a simple way by using the attribute `.parameters` or ```.named_parameters```

Let's illustrate this with the instance that we created above

```python
model = MNIST_FC_2(10,20)
for name, param in model.named_parameters():
    print(name)
    print(param)
```

<span style="color: grey;">Out:</span>
```text
box.0.weight
Parameter containing:
tensor([[ 0.0342, -0.0175,  0.0111,  ...,  0.0141,  0.0148,  0.0298],
        [ 0.0221, -0.0053, -0.0288,  ...,  0.0140,  0.0304, -0.0238],
        [ 0.0048,  0.0313, -0.0280,  ..., -0.0033,  0.0124, -0.0207],
        ...,
        [-0.0094, -0.0225,  0.0003,  ...,  0.0270,  0.0045,  0.0064],
        [ 0.0237, -0.0245,  0.0131,  ..., -0.0153,  0.0327, -0.0356],
        [-0.0160, -0.0242,  0.0298,  ..., -0.0164,  0.0119,  0.0063]],
       requires_grad=True)
box.0.bias
Parameter containing:
tensor([-0.0165,  0.0180,  0.0054, -0.0157,  0.0099,  0.0019, -0.0298, -0.0139,
        -0.0155,  0.0307], requires_grad=True)
box.2.weight
Parameter containing:
...
```

Alternatively, if we want to access parameters individually, we can usе an iterator with ```iter()```

```python
model = MNIST_FC_2(10,20)
iterator_params=iter(model.named_parameters())
name0, param0 = next(iterator_params)
name1, param1 = next(iterator_params)
...
print(param0.shape)
print(param0[:,2])
print(param1)
```

<span style="color: grey;">Out:</span>
```text
torch.Size([10, 784])
tensor([ 0.0111, -0.0288, -0.0280,  0.0103, -0.0095,  0.0346,  0.0215,  0.0003,
         0.0131,  0.0298], grad_fn=<SelectBackward0>)
Parameter containing:
tensor([-0.0165,  0.0180,  0.0054, -0.0157,  0.0099,  0.0019, -0.0298, -0.0139,
        -0.0155,  0.0307], requires_grad=True)
```


# Training

### Loss function

The whole idea of training is to find a fucntion (`model`) in a certain class (specific architecture) that maps the input to the prediction and minimizes a loss function, which in turn is defined as a "distance" between the prediction and the actual value(s).l

We have already chose the class of functions `class MNIST_FC_2L`. But there are various loss finctions

```python
nn.MSELoss()            # Mean Squared Error
nn.CrossEntropyLoss()   # Cross Entropy
nn.BCELoss()            # Binary Cross Entropy
...
```
These loss functions are defined as

```python
MSELoss()
```

\begin{equation}
J(\vec x,\vec y) = \frac{1}{N}(\vec x -\vec y)^2 
\end{equation}

```python
CrossEntropyLoss()
```

\begin{equation}
J(\vec x,\vec y) = - \frac{1}{N} \sum_{n=1}^N \sum_\alpha w_n y_{n,\alpha} \log \frac{e^{x_{n,\alpha}}}{\sum_\beta e^{x_{n,\beta}}}
\end{equation}

The arguments are as follows: $$y$$ is a `batch_size` collection of one-hot vectors of true labels, i.e., `y.shape = (num_classes, batch_size)`. $$x$$ is a vector of logits (unnormalized). To turn them in probabilities `CrossEntropyLoss()` runs itself the `softmax()` function by defining the corresponding probability
\begin{equation}
q_{n,\alpha} = \frac{e^{x_{n,\alpha}}}{\sum_\beta e^{x_{n,\beta}}}
\end{equation}

`BCELoss()` is defined similarly to `CrossEntropyLoss()` but for binary probabilities

\begin{equation}
J(\vec x,\vec y) = - \frac{1}{N} \sum_{n=1}^N w_n \Big [ y_n \log x_n + (1 - y_n) \log (1 - x_n) \Big ]
\end{equation}

It should be noted that in this definition 

\begin{equation}
\log x \geq -100 
\end{equation}

which allows to avoid invinite values of the log for $$x\to 0$$.

### Rational for choosing a loss function

In general, loss minimization of a loss function corresponds to the maximization of the likelihood. For instance, in the case of a classifier for `MNIST`, we observe a picture `X` of a hadwritten digit `\alpha` and its label `y`. We assume one-hot encoding, meaning that 

\begin{equation}
y_\beta = \delta_{\alpha\beta}
\end{equation}

Given the observation, we want to predict what digit this is. Our `model` only takes the picture as input and predicts the probability of a digit occurring given the observation, albeit partial, since we do not feed the label to the model. The model produces the conditional probability of observing a digit $$\beta$$ 

\begin{equation}
q^{(\theta)}_{\beta} \equiv \mathrm{Prob} _ \theta (Y=y|X)
\end{equation}

Therefore, the conditional probability of this specific observation, the likelihood, according to the model is

\begin{equation}
\ell(\theta|X,y) = \prod_{\beta} \Big [ q^{(\theta)}_{\beta} \Big ]^{y _ \beta} = 
 q^{(\theta)} _ {\alpha}
\end{equation}

We want to maximize the likelohood for the whole dataset, i.e.

\begin{equation}
L(\theta|X,y) = \prod_n\prod_{\beta} \Big [ q^{(\theta)}_{n,\beta} \Big ]^{y _ {n,\beta}}
\end{equation}

Computing $$- \log$$ results in the cross entropy loss function 


### Optimization

Now we need to optimize the chosen loss function

```python
J = nn.CrossEntropyLoss()
```

To do so, we use one of the optimizers provided by `torch`, for instance,

```python
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) # We need to specify the parameters the optimizer will be updating
```
The Stochastic Gradient Descent (SGD) corresponds to a classical particle moving in a potential $$U(x)$$

\begin{equation}
m \ddot {\vec x} + k \dot {\vec x} = - \vec \nabla U
\end{equation}

Goint to the first order (Hamiltonian) representation we get

\begin{equation}
\dot {\vec x} = v, \quad m \dot {\vec v} + k \vec v = - \vec \nabla U
\end{equation}


The discretized time version of this syste, with a step $$\varepsilon$$ is given by

\begin{equation}
\vec x_{t+1} = \vec x_t + \varepsilon \vec v_{t+1}, \quad \vec v_{t+1} = \vec v_t - \varepsilon \frac{k}{m} \vec v_t - \frac{\varepsilon}{m} \vec \nabla U (x_{t})
\end{equation}

Introducing the parameters 

\begin{equation}
\mu = 1 - \varepsilon \frac{k}{m}, \quad 1- \tau = \frac{\varepsilon}{m}, \quad \gamma = \varepsilon
\end{equation}

we arrive at the following rule for update

\begin{equation}
\vec x_{t+1} = \vec x_t + \gamma \vec v_{t+1}, \quad \vec v_{t+1} = \mu \vec v_t - ( 1 - \tau ) \vec \nabla U (x_{t})
\end{equation}

With all the components defined we run the following fro just one batch

```python
data_iterator = iter(train_dataloader)
X, y = next(data_iterator)
y_pred = model(X)
loss = J(y_pred, y)
loss.bachward()
optimizer.step()
optimizer.zero_grad()
```

Note that in the data `y` is not one-hot, nevertheless the `nn.CrossEntropyLoss()` works, since it can take the target in two different formats. It can be passed equivalently as (althogh we would be doing additional work in vain)

```python
y_hot = nn.functional.one_hot(y).float()
```
